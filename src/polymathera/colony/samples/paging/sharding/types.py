
import uuid
from pydantic import BaseModel, Field, ConfigDict, field_serializer


class ShardFileSegment(BaseModel):
    """Represents a segment of a file to be included in a shard"""

    file_path: str
    start_line: int
    end_line: int
    content: str | None = Field(default=None, description="None for binary files")
    language: str | None = Field(default=None, description="None for binary files")
    imports: set[str] = Field(default_factory=set, description="Imports in the segment")
    dependencies: set[str] = Field(default_factory=set, description="Dependencies in the segment")
    mime_type: str = Field(default="", description="MIME type of the segment's parent file (e.g. 'text/plain')")
    encoding: str = Field(default="", description="Encoding of the segment's parent file (e.g. 'utf-8')")
    is_binary: bool = Field(default=False, description="Whether the segment is a binary file")


class ShardMetadata(BaseModel):
    """Metadata about a repository shard"""

    # Ensures the model uses modern Pydantic v2 serialization behavior
    model_config = ConfigDict()

    shard_id: str
    file_segments: list[ShardFileSegment] = Field(default_factory=list)
    content_size_bytes: int = 0  # Size of actual content (excluding binary files)
    token_count: int = 0  # Actual token count from tokenizer (set during sharding)
    binary_files: set[str] = Field(default_factory=set)
    creation_timestamp: float = 0.0
    git_commit_hash: str = ""

    def intersects_file(self, file_path: str) -> bool:
        """Check if the shard intersects with a file"""
        return any(
            file_path == file_segment.file_path for file_segment in self.file_segments
        )

    @field_serializer('binary_files')
    def serialize_binary_files(self, binary_files: set[str], _info):
        return list(binary_files)


class RepositoryShard(BaseModel):
    """A shard of repository content with metadata"""
    shard_id: str = Field(description="The ID of the shard", default_factory=lambda: str(uuid.uuid4()))
    metadata: ShardMetadata
    annotated_content: str  # Content after prompt strategy processing
    raw_content: str  # Original concatenated content before annotation


class ShardingError(Exception):
    """Base class for all sharding-related errors"""
    pass


class SecurityError(Exception):
    """Exception raised for security-related issues"""
    pass

