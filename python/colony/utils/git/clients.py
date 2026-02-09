from __future__ import annotations
import base64
import requests
from typing import Any, ClassVar
import aiohttp
from pydantic import Field
from urllib.parse import urljoin

from ..config import ConfigComponent, register_polymathera_config



class GitClientBase:

    async def _fetch_data(self, url: str) -> dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                return await response.json()

    def _fetch_data_sync(self, url: str) -> dict[str, Any] | None:
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        return None



@register_polymathera_config()
class GitHubClientConfig(ConfigComponent):
    github_token: str = Field(default="", json_schema_extra={"env": "GITHUB_TOKEN"})

    CONFIG_PATH: ClassVar[str] = "gitutils.github_client"


class GitHubClient(GitClientBase):

    def __init__(self, config: GitHubClientConfig | None = None):
        self.config: GitHubClientConfig | None = config
        self.base_url: str | None = None
        self.headers: dict[str, str] | None = None

    async def initialize(self) -> None:
        self.config = await GitHubClientConfig.check_or_get_component(self.config)
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.config.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def get_file_contents_sync(self, repo: str, file_path: str) -> str | None:
        url = f"{self.base_url}repos/{repo}/contents/{file_path}"
        content = self._fetch_data_sync(url)
        if content and content.get("content"):
            return base64.b64decode(content["content"]).decode("utf-8")
        return None

    async def get_issues(self, owner: str, repo: str) -> list[dict[str, Any]]:
        url = f"{self.base_url}/repos/{owner}/{repo}/issues?state=all"
        return await self._fetch_data(url)

    async def get_pull_requests(self, owner: str, repo: str) -> list[dict[str, Any]]:
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls?state=all"
        return await self._fetch_data(url)

    async def get_comments(
        self, owner: str, repo: str, issue_number: int
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments"
        return await self._fetch_data(url)

    async def get_branches(self, owner: str, repo: str) -> list[dict[str, Any]]:
        url = f"{self.base_url}/repos/{owner}/{repo}/branches"
        return await self._fetch_data(url)

    async def get_tags(self, owner: str, repo: str) -> list[dict[str, Any]]:
        url = f"{self.base_url}/repos/{owner}/{repo}/tags"
        return await self._fetch_data(url)

    async def get_pull_request_reviews(
        self, owner: str, repo: str, pull_number: int
    ) -> list[dict[str, Any]]:
        """
        Fetch code reviews for a specific pull request.

        :param owner: The owner of the repository
        :param repo: The name of the repository
        :param pull_number: The number of the pull request
        :return: A list of code reviews for the specified pull request
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pull_number}/reviews"
        return await self._fetch_data(url)


@register_polymathera_config()
class GitLabClientConfig(ConfigComponent):
    gitlab_token: str = Field(default="", json_schema_extra={"env": "GITLAB_TOKEN"})
    gitlab_url: str = Field(
        default="https://gitlab.com",
        json_schema_extra={"env": "GITLAB_URL", "optional": True}
    )

    CONFIG_PATH: ClassVar[str] = "gitutils.gitlab_client"


class GitLabClient(GitClientBase):
    def __init__(self, config: GitLabClientConfig | None = None):
        self.config: GitLabClientConfig | None = config
        self.base_url: str | None = None
        self.headers: dict[str, str] | None = None

    async def initialize(self) -> None:
        self.config = await GitLabClientConfig.check_or_get_component(self.config)
        self.base_url = urljoin(self.config.gitlab_url, "/api/v4/")
        self.headers = {"Authorization": f"Bearer {self.config.gitlab_token}"}

    def get_file_contents_sync(self, project_id: int, file_path: str) -> str | None:
        url = f"{self.base_url}projects/{project_id}/repository/files/{file_path}/raw"
        content = self._fetch_data_sync(url)
        if content:
            return base64.b64decode(content["content"]).decode("utf-8")
        return None


