from rich.console import Console
from rich.syntax import Syntax
import subprocess


def print_sass(kernel, kernel_name=""):
    """Extract CUBIN from kernel and print SASS using rich

    Args:
        kernel: Triton kernel object
        kernel_name: Name of the kernel to use in the output
    """
    console = Console()

    # Write out the kernel cubin
    cubin_filename = f"{kernel_name}.cubin"
    with open(cubin_filename, "wb") as f:
        f.write(kernel.asm["cubin"])

    console.print(f"[green]✓[/green] Wrote CUBIN to {cubin_filename}")

    # Run cuobjdump to get SASS
    try:
        result = subprocess.run(
            ["cuobjdump", "--dump-sass", cubin_filename],
            capture_output=True,
            text=True,
            check=True,
        )

        sass_output = result.stdout

        # Print SASS using rich syntax highlighting
        console.print("\n[bold blue]SASS Disassembly:[/bold blue]")
        console.print("=" * 80)

        # Use assembly syntax highlighting
        syntax = Syntax(sass_output, "asm", theme="monokai", line_numbers=True)
        console.print(syntax)

        return sass_output

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running cuobjdump:[/red] {e}")
        console.print(f"[red]stderr:[/red] {e.stderr}")
        return None
    except FileNotFoundError:
        console.print(
            "[red]Error:[/red] cuobjdump not found. Make sure CUDA toolkit is installed and in PATH."
        )
        return None
