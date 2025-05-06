import gradio as gr
from datetime import datetime
from uk_economic_data.services.data_loader import DataLoader
from uk_economic_data.services.analysis_service import AnalysisService

class DashboardUI:
    def __init__(self):
        self.data_loader = DataLoader()
        self.analysis_service = AnalysisService()

    async def analyze_changes(
        self,
        start: str,
        end: str,
        n: int,
        min_age: int,
        max_age: int,
        threshold: int,
        period: str
    ) -> str:
        try:
            # Validate date range
            min_date = datetime(2025, 3, 1)
            max_date = datetime(2025, 4, 30)
            start_date = datetime.strptime(start, "%Y-%m-%d")
            end_date = datetime.strptime(end, "%Y-%m-%d")
            
            if start_date < min_date or end_date > max_date:
                return "Date range must be between 2025-03-01 and 2025-04-30"
            
            # Load and preprocess data
            df = self.data_loader.load_data()
            if df is None:
                return "Failed to load data"
                
            df = self.data_loader.preprocess_data(df)
            if df is None:
                return "Failed to preprocess data"
            
            # Filter by date range
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # Get top N products with insights
            products, insights = await self.analysis_service.get_top_products_with_insights(
                df=df,
                top_n=n,
                min_age=min_age,
                max_age=max_age,
                threshold=threshold,
                period=period
            )
            
            # Format results with Markdown
            result_text = f"# 🔍 Top {n} Products with Largest Sales Fluctuations\n\n"
            result_text += f"### 👥 Target Age Group: {min_age}-{max_age}\n"
            result_text += f"### 📉 Alert Threshold: {threshold}% decrease\n"
            result_text += f"### 📅 Comparison Period: {period}\n\n"
            
            for product in products:
                result_text += f"## 📦 {product}\n\n"
                for insight in insights[product]:
                    if insight.startswith("📌 Cause Analysis:"):
                        result_text += f"### {insight}\n\n"
                    elif insight.startswith("💡 Strategy Suggestions:"):
                        result_text += f"### {insight}\n\n"
                    else:
                        result_text += f"{insight}\n\n"
                result_text += "---\n\n"
            
            return result_text
            
        except Exception as e:
            return f"Error: {str(e)}"

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface."""
        with gr.Blocks(
            title="Retail Sales Analysis Dashboard",
            theme=gr.themes.Soft()
        ) as demo:
            gr.Markdown("# Retail Sales Analysis Dashboard")
            gr.Markdown("Analyze sales changes and get actionable marketing insights for your products.")

            with gr.Row():
                with gr.Column(scale=1):
                    # Basic analysis settings
                    gr.Markdown("### 📅 Analysis Period")
                    start_date = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)",
                        value="2025-03-01",
                        placeholder="YYYY-MM-DD",
                    )
                    end_date = gr.Textbox(
                        label="End Date (YYYY-MM-DD)",
                        value="2025-04-30",
                        placeholder="YYYY-MM-DD",
                    )
                    top_n = gr.Radio(
                        choices=[5, 10],
                        value=5,
                        label="Number of Products to Analyze",
                        info="Select how many top products to analyze"
                    )

                    # Alert settings
                    gr.Markdown("### 🔔 Alert Settings")
                    with gr.Row():
                        min_age = gr.Slider(
                            minimum=18,
                            maximum=65,
                            value=18,
                            step=1,
                            label="Minimum Age"
                        )
                        max_age = gr.Slider(
                            minimum=18,
                            maximum=65,
                            value=35,
                            step=1,
                            label="Maximum Age"
                        )
                    decrease_threshold = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=5,
                        step=1,
                        label="Decrease Threshold (%)",
                        info="Alert when sales decrease by this percentage"
                    )
                    comparison_period = gr.Radio(
                        choices=["daily", "weekly", "monthly"],
                        value="weekly",
                        label="Comparison Period",
                        info="Compare sales over this period"
                    )
                    
                    analyze_btn = gr.Button("Analyze Data", variant="primary")

                with gr.Column(scale=2):
                    output = gr.Markdown(
                        label="Analysis Results",
                        container=True,
                        show_label=True,
                        elem_classes=["analysis-results"]
                    )

            analyze_btn.click(
                fn=self.analyze_changes,
                inputs=[
                    start_date, 
                    end_date, 
                    top_n,
                    min_age,
                    max_age,
                    decrease_threshold,
                    comparison_period
                ],
                outputs=output,
            )

        return demo 