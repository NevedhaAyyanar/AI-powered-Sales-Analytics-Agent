# Gradio ChatInterface entry point
import gradio as gr
from src.agent import run_agent


def chat(user_message: str, history: list) -> str:
    """Handle chat messages from Gradio."""
    chat_history = []
    for msg in history:
        if isinstance(msg, dict):
            chat_history.append({"role": msg["role"], "content": msg["content"]})
        elif isinstance(msg, (list,tuple)) and len(msg) == 2:
            chat_history.append({"role": "user", "content": msg[0]})
            chat_history.append({"role": "assistant", "content": msg[1]})
            
    return run_agent(user_message, chat_history)


demo = gr.ChatInterface(
    fn=chat,
    title="FMCG Sales Analytics Agent",
    description="AI-powered sales data analysis for February 2025. Ask questions about revenue, trends, products, customers, and data quality.",
    examples=[
        "Load sales data for February 1, 2025 and profile it",
        "Load data for the first week of February and analyze revenue trends",
        "Show me the top products by revenue for Feb 1-7",
        "Check data quality for February 1, 2025",
        "Analyze customer settlement patterns for the full month",
    ]
    
)

if __name__ == "__main__":
    demo.launch(share=True)