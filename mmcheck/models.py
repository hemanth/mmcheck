"""ModelInfo dataclass for representing model capabilities."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelInfo:
    """Represents a model's multimodal capabilities."""

    name: str
    multimodal: bool = False
    input_modalities: List[str] = field(default_factory=lambda: ["text"])
    output_modalities: List[str] = field(default_factory=lambda: ["text"])
    architecture: Optional[str] = None
    source: str = "unknown"
    model_type: Optional[str] = None
    parameters: Optional[str] = None

    def supports(self, modality: str) -> bool:
        """Check if the model supports a given input modality."""
        return modality.lower() in [m.lower() for m in self.input_modalities]

    def supports_output(self, modality: str) -> bool:
        """Check if the model supports a given output modality."""
        return modality.lower() in [m.lower() for m in self.output_modalities]

    @property
    def input_str(self) -> str:
        return ", ".join(self.input_modalities)

    @property
    def output_str(self) -> str:
        return ", ".join(self.output_modalities)

    def __str__(self) -> str:
        mm = "Yes" if self.multimodal else "No"
        lines = [
            f"Model:      {self.name}",
            f"Multimodal: {mm}",
            f"Inputs:     {self.input_str}",
            f"Outputs:    {self.output_str}",
        ]
        if self.architecture:
            lines.append(f"Arch:       {self.architecture}")
        if self.model_type:
            lines.append(f"Type:       {self.model_type}")
        lines.append(f"Source:     {self.source}")
        return "\n".join(lines)
