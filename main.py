from src.data_pipeline import generate_synthetic_sales_data
from src.predictive_analytics import train_predictive_model
from src.neural_forecast import train_neural_model
from src.agentic_orchestrator import run_agentic_workflow


def main():
    print("=" * 90)
    print("AI ANALYTICS AGENTIC SYSTEM")
    print("Predictive Analytics + Neural Networks + LangChain + Agentic AI")
    print("=" * 90)

    sales_df = generate_synthetic_sales_data(days=365)

    predictive_report = train_predictive_model(sales_df)
    neural_report = train_neural_model(sales_df)
    agent_report = run_agentic_workflow(predictive_report, neural_report, sales_df)

    print("\n[1] PREDICTIVE ANALYTICS REPORT")
    print(f"R2 Score: {predictive_report['metrics']['r2']:.4f}")
    print(f"MAE: {predictive_report['metrics']['mae']:.4f}")
    print(f"RMSE: {predictive_report['metrics']['rmse']:.4f}")
    print(f"Next 7 day forecast: {predictive_report['forecast']}")

    print("\n[2] NEURAL NETWORK REPORT")
    print(f"Neural MAE: {neural_report['metrics']['mae']:.4f}")
    print(f"Neural RMSE: {neural_report['metrics']['rmse']:.4f}")
    print(f"Neural next 7 day forecast: {neural_report['forecast']}")

    print("\n[3] LANGCHAIN-STYLE AI INSIGHT")
    print(agent_report['insight'])

    print("\n[4] AGENTIC EXECUTION SUMMARY")
    for task in agent_report['tasks']:
        print(f"- {task}")

    print("\n[5] ACTIONABLE RECOMMENDATIONS")
    for recommendation in agent_report['recommendations']:
        print(f"- {recommendation}")

    print("\n" + "=" * 90)
    print("SYSTEM EXECUTION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
