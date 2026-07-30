import json
import os
import sys

import requests
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.text import Text

from src.api.schemas import UserRequest, UserResponse
from src.generation.schemas import BaseResponseSchema


def main():
    api_host = os.getenv("API_HOST", default="127.0.0.1")
    api_port = int(os.getenv("API_PORT", default="8080"))
    api_path = f"http://{api_host}:{api_port}/generate"
    console = Console()

    console.print(
        Panel(
            "[bold cyan]RAG Inference Interface[/bold cyan]\n"
            "[dim]Type your query and press Enter. Type 'exit' or 'quit' to close.[/dim]",
            title="[bold magenta]FastAPI Client[/bold magenta]",
            border_style="magenta",
            expand=False,
        )
    )

    while True:
        try:
            # Use Rich Prompt for user input with distinct styling
            user_query = Prompt.ask("\n[bold green]You[/bold green]").strip()

            # Check for exit condition
            if user_query.lower() in ["exit", "quit"]:
                console.print("\n[bold yellow]Goodbye![/bold yellow] 👋")
                sys.exit(0)

            if not user_query:
                continue

            # Create a localized status spinner while waiting for FastAPI
            with Status(
                "[bold blue]Thinking...[/bold blue]", spinner="dots", console=console
            ):
                payload = UserRequest(prompt=user_query)
                response = requests.post(
                    api_path, json=payload.model_dump(), timeout=60
                )

            # Process and display the response
            if response.status_code == 200:
                data = response.json()
                user_response = UserResponse.model_validate(data)
                status = user_response.status
                if status == 200:
                    rag_response = BaseResponseSchema.model_validate(
                        json.loads(user_response.generated_response)
                    )
                    # Render the response inside a clean border panel
                    thought_panel = Panel(
                        Text(rag_response.thought_process, style="white"),
                        title="[bold red]Thought Process[/bold red]",
                        border_style="red",
                        expand=False,
                    )
                    answer_panel = Panel(
                        Text(rag_response.answer, style="white"),
                        title="[bold green]Answer[/bold green]",
                        border_style="green",
                        expand=False,
                    )
                    sources_panel = Panel(
                        Text(", ".join(rag_response.sources), style="white"),
                        title="[bold yellow]Sources[/bold yellow]",
                        border_style="yellow",
                        expand=False,
                    )
                    inner_group = Group(thought_panel, answer_panel, sources_panel)
                    outer_panel = Panel(
                        inner_group,
                        title="[bold blue]System Response[/bold blue]",
                        border_style="blue",
                        expand=False,
                    )
                    console.print(outer_panel)
                else:
                    console.print(
                        f"[bold red]Error:[/bold red] Server responded with status code {status}, detail: {user_response.generated_response}",
                        style="red",
                    )
            else:
                console.print(
                    f"[bold red]Error:[/bold red] Server responded with status code {response.status_code}",
                    style="red",
                )

        except requests.exceptions.ConnectionError:
            console.print(
                "\n[bold red]Error:[/bold red] Could not connect to the FastAPI server. Ensure it is running.",
                style="red",
            )
        except KeyboardInterrupt:
            console.print(
                "\n\n[bold yellow]Session interrupted. Goodbye![/bold yellow] 👋"
            )
            sys.exit(0)


if __name__ == "__main__":
    main()  # pragma: no cover
