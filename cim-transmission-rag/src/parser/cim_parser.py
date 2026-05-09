"""
CIM XML Parser for Transmission System RAG
Parses IEC 61970/61968 CIM/XML files and produces structured text chunks
suitable for embedding and retrieval.
"""

from rdflib import Graph, Namespace, RDF, RDFS
from rdflib.namespace import NamespaceManager
from typing import Generator
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

CIM = Namespace("http://iec.ch/TC57/2013/CIM-schema-cim16#")

# CIM classes relevant to transmission systems
TRANSMISSION_CLASSES = [
    "Substation",
    "ACLineSegment",
    "PowerTransformer",
    "PowerTransformerEnd",
    "Breaker",
    "Disconnector",
    "BusbarSection",
    "LinearShuntCompensator",
    "SeriesCompensator",
    "VoltageLevel",
    "Bay",
    "BaseVoltage",
    "ConnectivityNode",
    "Terminal",
    "GeographicalRegion",
    "SubGeographicalRegion",
    "GeneratingUnit",
    "SynchronousMachine",
    "ExternalNetworkInjection",
]

# Human-readable field descriptions for better RAG quality
FIELD_LABELS = {
    "IdentifiedObject.name": "Name",
    "IdentifiedObject.description": "Description",
    "IdentifiedObject.mRID": "Unique ID",
    "Conductor.length": "Length (km)",
    "ACLineSegment.r": "Positive-sequence resistance (pu)",
    "ACLineSegment.x": "Positive-sequence reactance (pu)",
    "ACLineSegment.bch": "Positive-sequence charging susceptance (pu)",
    "ACLineSegment.r0": "Zero-sequence resistance (pu)",
    "ACLineSegment.x0": "Zero-sequence reactance (pu)",
    "BaseVoltage.nominalVoltage": "Nominal voltage (kV)",
    "PowerTransformerEnd.ratedU": "Rated voltage (kV)",
    "PowerTransformerEnd.ratedS": "Rated power (MVA)",
    "PowerTransformerEnd.r": "Resistance (pu)",
    "PowerTransformerEnd.x": "Reactance (pu)",
    "PowerTransformerEnd.connectionKind": "Winding connection",
    "PowerTransformerEnd.endNumber": "End number",
    "Switch.normalOpen": "Normally open",
    "Switch.ratedCurrent": "Rated current (A)",
    "ShuntCompensator.nomU": "Nominal voltage (kV)",
    "LinearShuntCompensator.bPerSection": "Susceptance per section (pu)",
    "ShuntCompensator.maximumSections": "Maximum sections",
    "ShuntCompensator.normalSections": "Normal sections",
    "BusbarSection.ipMax": "Max short-circuit current (kA)",
}


@dataclass
class CIMChunk:
    """A single CIM object represented as a text chunk for embedding."""
    chunk_id: str
    cim_class: str
    object_id: str
    name: str
    text: str
    metadata: dict = field(default_factory=dict)


class CIMParser:
    """
    Parses CIM/XML (RDF/XML serialization of IEC 61970) files.
    Produces CIMChunk objects ready for embedding.
    """

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.graph = Graph()
        self._load()

    def _load(self):
        logger.info(f"Loading CIM file: {self.xml_path}")
        self.graph.parse(self.xml_path, format="xml")
        logger.info(f"Loaded {len(self.graph)} triples")

    def _short_id(self, uri: str) -> str:
        """Extract the local name from a URI."""
        if "#" in uri:
            return uri.split("#")[-1]
        return uri.split("/")[-1]

    def _get_field(self, subject, property_local_name: str) -> str | None:
        """Get a CIM property value as a string."""
        prop = CIM[property_local_name]
        val = self.graph.value(subject, prop)
        return str(val) if val else None

    def _build_chunk_text(self, subject, cim_class: str, props: dict) -> str:
        """
        Build a rich natural-language text chunk from a CIM object.
        This format is optimised for retrieval — reads like a datasheet.
        """
        name = props.get("IdentifiedObject.name", self._short_id(str(subject)))
        description = props.get("IdentifiedObject.description", "")

        lines = [
            f"CIM Class: {cim_class}",
            f"Asset Name: {name}",
        ]
        if description:
            lines.append(f"Description: {description}")

        lines.append("")
        lines.append("Technical parameters:")

        for prop_key, value in props.items():
            if prop_key in ("IdentifiedObject.name", "IdentifiedObject.description"):
                continue
            label = FIELD_LABELS.get(prop_key, prop_key.split(".")[-1])
            lines.append(f"  - {label}: {value}")

        # Add linked objects (relationships)
        for pred, obj in self.graph.predicate_objects(subject):
            pred_local = self._short_id(str(pred))
            if pred_local.endswith("resource") or str(pred).startswith(str(RDF)):
                continue
            if pred_local not in FIELD_LABELS and "." in pred_local:
                obj_local = self._short_id(str(obj))
                # Only include references to other CIM objects (not literals)
                if obj_local.startswith(tuple(c[:3] for c in TRANSMISSION_CLASSES)):
                    lines.append(f"  - {pred_local}: {obj_local}")

        return "\n".join(lines)

    def parse(self) -> Generator[CIMChunk, None, None]:
        """
        Parse all transmission-relevant CIM objects into CIMChunks.
        Yields one chunk per CIM object instance.
        """
        for cim_class_name in TRANSMISSION_CLASSES:
            cim_type = CIM[cim_class_name]
            instances = list(self.graph.subjects(RDF.type, cim_type))

            if not instances:
                continue

            logger.info(f"Found {len(instances)} {cim_class_name} instances")

            for subject in instances:
                props = {}

                for pred, obj in self.graph.predicate_objects(subject):
                    pred_local = self._short_id(str(pred))
                    # Collect only CIM datatype properties (literals)
                    if hasattr(obj, "toPython"):
                        props[pred_local] = str(obj.toPython())
                    elif pred_local in FIELD_LABELS:
                        props[pred_local] = self._short_id(str(obj))

                object_id = self._short_id(str(subject))
                name = props.get("IdentifiedObject.name", object_id)
                text = self._build_chunk_text(subject, cim_class_name, props)

                chunk = CIMChunk(
                    chunk_id=f"{cim_class_name}_{object_id}",
                    cim_class=cim_class_name,
                    object_id=object_id,
                    name=name,
                    text=text,
                    metadata={
                        "cim_class": cim_class_name,
                        "object_id": object_id,
                        "name": name,
                        "source_file": self.xml_path,
                    },
                )
                yield chunk


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)
    parser = CIMParser("../../data/raw/sample_transmission.xml")
    chunks = list(parser.parse())

    print(f"\nParsed {len(chunks)} CIM chunks\n")
    for chunk in chunks[:3]:
        print("=" * 60)
        print(chunk.text)
        print()
