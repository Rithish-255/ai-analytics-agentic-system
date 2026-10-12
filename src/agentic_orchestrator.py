from src.langchain_agent import LangChainAnalyticsAgent, AgenticPlanner


def run_agentic_workflow(predictive_report: dict, neural_report: dict, sales_df):
    planner = AgenticPlanner()
    agent = LangChainAnalyticsAgent()

    tasks = planner.plan_tasks(predictive_report, neural_report)
    summary = {
        "avg_sales": float(sales_df["sales"].mean()),
        "peak_sales": float(sales_df["sales"].max()),
        "trend_direction": "upward" if sales_df["sales"].iloc[-1] > sales_df["sales"].mean() else "stable",
    }

    insight = agent.generate_insight(predictive_report, neural_report, summary)
    recommendations = [
        "Increase marketing spend during peak demand windows to capture demand lift.",
        "Balance pricing against demand elasticity to avoid revenue erosion.",
        "Use forecast-driven inventory planning to reduce stockouts and overstock.",
        "Monitor weather and promotion impacts to improve model precision.",
    ]

    return {
        "tasks": tasks,
        "insight": insight,
        "recommendations": recommendations,
    }
