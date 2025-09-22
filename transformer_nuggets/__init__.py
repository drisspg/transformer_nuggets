from transformer_nuggets import quant as quant, utils as utils, numerics as numerics
import logging


def init_logging(level=logging.INFO):
    """
    Configure logging for transformer_nuggets library at INFO level.
    Adds a StreamHandler if none exists.
    """
    import logging

    logger = logging.getLogger("transformer_nuggets")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False


def vscode_breakpoint(port=5691):
    """
    Set up a VSCode debugger breakpoint for remote debugging.

    This function configures the debugpy module to listen for a VSCode debugger
    connection on the specified port, provides clear instructions for manual attachment,
    waits for the debugger to attach, and then sets a breakpoint at the current location.

    Args:
        port (int, optional): The port number to listen for debugger connections.
                             Defaults to 5678.

    Returns:
        None

    Example:
        >>> vscode_breakpoint()  # Uses default port 5678
        >>> vscode_breakpoint(9999)  # Uses custom port 9999

    Note:
        - Requires the 'debugpy' package to be installed
        - Provides clear instructions for VS Code debugger attachment
        - The function will block execution until a debugger is attached
        - Typically used during development for remote debugging scenarios
    """
    import debugpy

    if not debugpy.is_client_connected():
        debugpy.listen(port)

        from rich.console import Console

        console = Console()
        console.print(f"🐛 [bold green]Debugger ready on port {port}[/bold green]")
        console.print("📋 [yellow]In VS Code: Press F5 -> 'Attach to Python Debugger'[/yellow]")

        debugpy.wait_for_client()
    debugpy.breakpoint()
