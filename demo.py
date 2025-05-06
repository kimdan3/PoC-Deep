import os
from datetime import datetime
import asyncio
import gradio as gr
from dotenv import load_dotenv
from uk_economic_data import UKEconomicDataFetcher
from uk_economic_data.services.data_loader import DataLoader
from uk_economic_data.services.analysis_service import AnalysisService

from config import DATE_CONFIG, UI_CONFIG, ANALYSIS_CONFIG
from utils.logger import app_logger

# ──────────────────────────────────────────────────────────────────────────────
# Environment Configuration
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()

# Initialize services
uk_data_fetcher = UKEconomicDataFetcher()
data_loader = DataLoader()
analysis_service = AnalysisService()

# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI Construction
# ──────────────────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        title="Retail sales analysis dashboard",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# Retail sales analysis dashboard")
        gr.Markdown("Analyse sales changes and get actionable marketing insights for your products.")

        with gr.Row():
            with gr.Column(scale=1):
                # Alert settings
                gr.Markdown("### 🔔 Alert settings")
                with gr.Row():
                    gender = gr.Radio(
                        choices=ANALYSIS_CONFIG["GENDER_CHOICES"],
                        value=ANALYSIS_CONFIG["DEFAULT_GENDER"],
                        label="Gender",
                        info="Select target gender"
                    )
                with gr.Row():
                    min_age = gr.Slider(
                        minimum=UI_CONFIG["AGE"]["MIN"],
                        maximum=UI_CONFIG["AGE"]["MAX"],
                        value=UI_CONFIG["AGE"]["DEFAULT_MIN"],
                        step=1,
                        label="Minimum age"
                    )
                    max_age = gr.Slider(
                        minimum=UI_CONFIG["AGE"]["MIN"],
                        maximum=UI_CONFIG["AGE"]["MAX"],
                        value=UI_CONFIG["AGE"]["DEFAULT_MAX"],
                        step=1,
                        label="Maximum age"
                    )
                decrease_threshold = gr.Slider(
                    minimum=UI_CONFIG["THRESHOLD"]["MIN"],
                    maximum=UI_CONFIG["THRESHOLD"]["MAX"],
                    value=UI_CONFIG["THRESHOLD"]["DEFAULT"],
                    step=1,
                    label="Decrease threshold (%)",
                    info="Alert when sales decrease by this percentage"
                )
                
                # Comparison period
                gr.Markdown("#### 📊 Comparison Period")
                comparison_period = gr.Radio(
                    choices=ANALYSIS_CONFIG["PERIOD_CHOICES"],
                    value=ANALYSIS_CONFIG["DEFAULT_PERIOD"],
                    label="Current period analysis",
                    info="Compare sales over this period from current date"
                )
                
                # Custom date range analysis
                gr.Markdown("#### 📅 Custom date range analysis")
                with gr.Row():
                    start_date = gr.Textbox(
                        label="Start date",
                        value=DATE_CONFIG["MIN_DATE"].strftime(DATE_CONFIG["DATE_FORMAT"]),
                        placeholder=DATE_CONFIG["DATE_FORMAT"],
                        info="Format: YYYY-MM-DD"
                    )
                    end_date = gr.Textbox(
                        label="End date",
                        value=DATE_CONFIG["MAX_DATE"].strftime(DATE_CONFIG["DATE_FORMAT"]),
                        placeholder=DATE_CONFIG["DATE_FORMAT"],
                        info="Format: YYYY-MM-DD"
                    )
                
                with gr.Row():
                    top_n = gr.Dropdown(
                        choices=UI_CONFIG["TOP_N"]["CHOICES"],
                        value=str(UI_CONFIG["TOP_N"]["DEFAULT"]),
                        label="Number of products to analyse",
                        info="Select or enter custom number of products to analyse"
                    )
                    custom_top_n = gr.Number(
                        label="Custom number",
                        value=UI_CONFIG["TOP_N"]["DEFAULT"],
                        minimum=UI_CONFIG["TOP_N"]["MIN"],
                        maximum=UI_CONFIG["TOP_N"]["MAX"],
                        step=1,
                        visible=False
                    )
                
                analyze_btn = gr.Button("Analyse Data", variant="primary")

            with gr.Column(scale=2):
                output = gr.Markdown(
                    label="Analysis results",
                    container=True,
                    show_label=True,
                    elem_classes=["analysis-results"]
                )

        def update_custom_top_n_visibility(choice):
            return gr.update(visible=choice == "Custom")
        
        def validate_and_get_top_n(choice, custom_value):
            try:
                if choice == "Custom":
                    if not custom_value or custom_value < UI_CONFIG["TOP_N"]["MIN"] or custom_value > UI_CONFIG["TOP_N"]["MAX"]:
                        raise ValueError(f"Custom number must be between {UI_CONFIG['TOP_N']['MIN']} and {UI_CONFIG['TOP_N']['MAX']}")
                    return int(custom_value)
                return int(choice)
            except (ValueError, TypeError) as e:
                app_logger.error(f"Invalid number of products: {e}")
                raise gr.Error(f"Invalid number of products: {str(e)}")

        def validate_dates(start, end):
            try:
                start_date = datetime.strptime(start, DATE_CONFIG["DATE_FORMAT"])
                end_date = datetime.strptime(end, DATE_CONFIG["DATE_FORMAT"])
                
                if start_date > end_date:
                    raise ValueError("Start date must be before end date")
                
                if start_date < DATE_CONFIG["MIN_DATE"] or end_date > DATE_CONFIG["MAX_DATE"]:
                    raise ValueError(f"Date range must be between {DATE_CONFIG['MIN_DATE'].strftime(DATE_CONFIG['DATE_FORMAT'])} and {DATE_CONFIG['MAX_DATE'].strftime(DATE_CONFIG['DATE_FORMAT'])}")
                
                return start_date, end_date
            except ValueError as e:
                app_logger.error(f"Invalid date format: {e}")
                raise gr.Error(f"Invalid date format. Please use {DATE_CONFIG['DATE_FORMAT']} format")
            except Exception as e:
                app_logger.error(f"Invalid date range: {e}")
                raise gr.Error(f"Invalid date range: {str(e)}")

        async def analyze_with_validation(start, end, top_n_choice, custom_n, min_age, max_age, gender, threshold, period):
            try:
                app_logger.info("Starting analysis with validation")
                
                # Date validation
                start_date, end_date = validate_dates(start, end)
                
                # Top N validation
                n_products = validate_and_get_top_n(top_n_choice, custom_n)
                
                # Age range validation
                if min_age >= max_age:
                    raise gr.Error("Minimum age must be less than maximum age")
                
                app_logger.info(f"Analysis parameters: start_date={start_date}, end_date={end_date}, n_products={n_products}")
                
                # Load and preprocess data
                df = data_loader.load_data()
                if df is None:
                    return "Failed to load data"
                    
                df = data_loader.preprocess_data(df)
                if df is None:
                    return "Failed to preprocess data"
                
                # Filter by date range
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                # Get top N products with insights
                products, insights = await analysis_service.get_top_products_with_insights(
                    df=df,
                    gender=gender,
                    min_age=min_age,
                    max_age=max_age,
                    top_n=n_products,
                    threshold=threshold,
                    period=period
                )
                
                # Format results with Markdown
                result_text = f"# 🔍 Top {n_products} Products with largest sales fluctuations\n\n"
                result_text += f"### 👥 Target gender: {gender}\n"
                result_text += f"### 👥 Target age group: {min_age}-{max_age}\n"
                result_text += f"### 📉 Alert threshold: {threshold}% decrease\n"
                result_text += f"### 📅 Comparison period: {period}\n\n"
                
                for product in products:
                    result_text += f"## 📦 {product}\n\n"
                    for insight in insights[product]:
                        if insight.startswith("📌 Cause analysis:"):
                            result_text += f"### {insight}\n\n"
                        elif insight.startswith("💡 Strategy suggestions:"):
                            result_text += f"### {insight}\n\n"
                        else:
                            result_text += f"{insight}\n\n"
                    result_text += "---\n\n"
                
                return result_text
                
            except Exception as e:
                app_logger.error(f"Analysis error: {e}")
                if isinstance(e, gr.Error):
                    raise e
                raise gr.Error(f"An error occurred: {str(e)}")

        top_n.change(
            fn=update_custom_top_n_visibility,
            inputs=[top_n],
            outputs=[custom_top_n]
        )

        analyze_btn.click(
            fn=analyze_with_validation,
            inputs=[
                start_date,
                end_date,
                top_n,
                custom_top_n,
                min_age,
                max_age,
                gender,
                decrease_threshold,
                comparison_period
            ],
            outputs=output,
        )

    return demo

# ──────────────────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    try:
        app_logger.info("Starting application")
        demo = build_demo()
        app_logger.info("Launching Gradio interface")
        demo.launch(share=True)
    except Exception as e:
        app_logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
