import os

def detect_environment():
    """
    Detects the current Python environment: Jupyter, VS Code, Colab, or Other.
    Returns one of: 'jupyter', 'colab', 'vscode', 'other'
    """
    # Colab detection
    try:
        import google.colab  # type: ignore[import-not-accessed]
        return 'colab'
    except ImportError:
        pass
    # Jupyter detection
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]
        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ in ["ZMQInteractiveShell", "TerminalInteractiveShell"]:
            return 'jupyter'
    except Exception:
        pass
    # VS Code detection
    if os.environ.get("VSCODE_PID"):
        return 'vscode'
    return 'other'


def show_or_save_plotly_figure(fig, output_file):
    """
    Shows or saves a Plotly figure based on the detected environment.
    - Jupyter/Colab: fig.show()
    - VS Code: save HTML and open in browser
    - Other: save HTML only
    """
    env = detect_environment()
    if env in ['jupyter', 'colab']:
        fig.show()
    else:
        fig.write_html(output_file)
        print(f"3D Sweep Chart saved to {output_file}")
        if env == 'vscode':
            try:
                import webbrowser
                webbrowser.open(output_file)
            except Exception:
                pass
