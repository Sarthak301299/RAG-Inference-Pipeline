import json
import os
import sys

import requests
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.text import Text

from src.api.schemas import AgentQueryResponse, UserRequest, UserResponse
from src.pipeline.schemas import BaseResponseSchema


def main():
    api_host = os.getenv("API_HOST", default="127.0.0.1")
    api_port = int(os.getenv("API_PORT", default="8080"))
    mode = os.getenv("API_EXECUTION_MODE", default="agent")
    if mode not in ("agent", "rag"):
        raise ValueError("API_EXECUTION_MODE must be set to 'agent' or 'rag'.")
    api_path = f"http://{api_host}:{api_port}/generate"
    console = Console()

    if mode == "agent":
        console.print(
            Panel(
                "[bold cyan]Agentic Inference Interface[/bold cyan]\n"
                "[dim]Type your query and press Enter. Type 'exit' or 'quit' to close.[/dim]",
                title="[bold magenta]FastAPI Client[/bold magenta]",
                border_style="magenta",
                expand=False,
            )
        )
    else:
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
                    if mode == "agent":
                        agent_response = AgentQueryResponse.model_validate(
                            json.loads(user_response.generated_response)
                        )
                        step_panels = []
                        for index, step in enumerate(
                            agent_response.scratchpad, start=1
                        ):
                            step_content = (
                                f"[bold red]Thought:[/bold red] {step.thought}\n"
                                f"[bold cyan]Action:[/bold cyan] {step.action}\n"
                                f"[bold blue]Action Input:[/bold blue] {step.action_input}\n"
                                f"[bold green]Observation:[/bold green] {step.observation}"
                            )
                            step_panels.append(
                                Panel(
                                    step_content,
                                    title=f"Step {index}",
                                    border_style="dim",
                                )
                            )
                        summary_content = (
                            f"[bold magenta]Total Iterations Used:[/bold magenta] {agent_response.iterations_used}\n\n"
                            f"[bold green]Final Answer:[/bold green] {agent_response.answer}\n"
                        )
                        dashboard_content = Group(
                            Panel(
                                summary_content,
                                title="Summary Execution Info",
                                border_style="cyan",
                            ),
                            *step_panels,
                        )
                        console.print(
                            Panel(
                                dashboard_content,
                                title="Agent Execution Trace",
                                expand=False,
                            )
                        )
                    else:
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
