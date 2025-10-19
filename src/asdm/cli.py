# src/asdm/cli.py

"""
ASDM Command Line Interface

Main entry point for the unified ASDM CLI tool.
"""

import sys
import argparse
from asdm import __version__


def create_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='asdm',
        description='ASDM - A Python package for System Dynamics Modeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  asdm simulator              Launch the web-based simulator
  asdm simulator --port 9000  Launch simulator on custom port
  asdm --version              Show version information

For more information: https://github.com/wzh1895/ASDM
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'ASDM v{__version__}'
    )
    
    # Create subparsers for subcommands
    subparsers = parser.add_subparsers(
        title='Available commands',
        dest='command',
        help='Command to execute',
        metavar='<command>'
    )
    
    # Add simulator subcommand
    simulator_parser = subparsers.add_parser(
        'simulator',
        help='Launch the ASDM web-based simulator',
        description='Start a local web server with an interactive System Dynamics model simulator.'
    )
    
    simulator_parser.add_argument(
        'model_file',
        nargs='?',
        default=None,
        help='Optional: Path to model file (.stmx or .xmile) to load and run automatically'
    )
    
    simulator_parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host/IP address to bind to (default: 127.0.0.1)'
    )
    
    simulator_parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to run the server on (default: 8080)'
    )
    
    # Future subcommands can be added here:
    # run_parser = subparsers.add_parser('run', help='Run a model from command line')
    # optimize_parser = subparsers.add_parser('optimize', help='Run model optimization')
    # etc.
    
    return parser


def cmd_simulator(args):
    """Handle the 'simulator' subcommand."""
    from asdm.simulator.app import run_simulator
    run_simulator(args.host, args.port, args.model_file)


def main():
    """Main entry point for the unified ASDM CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # If no command specified, show help
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to appropriate command handler
    if args.command == 'simulator':
        cmd_simulator(args)
    else:
        # This shouldn't happen given the subparsers, but just in case
        parser.print_help()
        sys.exit(1)


def main_legacy():
    """
    Legacy entry point for 'asdm.simulator' command.
    Shows deprecation warning and redirects to new command.
    """
    import warnings
    
    # Show deprecation warning
    print("=" * 70)
    print("⚠️  DEPRECATION WARNING")
    print("=" * 70)
    print("The command 'asdm.simulator' is deprecated and will be removed soon.")
    print("Please use the new command instead:")
    print()
    print("  New:  asdm simulator")
    print("  Old:  asdm.simulator  (this command)")
    print()
    print("All functionality remains the same, just the command name changes.")
    print("=" * 70)
    print()
    
    # Run the simulator with the legacy entry point
    from asdm.simulator.app import main as legacy_main
    legacy_main()


if __name__ == '__main__':
    main()

