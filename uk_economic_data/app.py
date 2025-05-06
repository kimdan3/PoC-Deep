import gradio as gr
from uk_economic_data.services.retail_analysis import RetailAnalysisService
from uk_economic_data.models.retail_sales import RetailSalesResponse

def create_retail_analysis_interface():
    retail_service = RetailAnalysisService()
    
    def analyze_sales(data):
        try:
            # Convert data format
            retail_data = RetailSalesResponse(data)
            result = retail_service.analyze_retail_sales(retail_data)
            return result
        except Exception as e:
            return f"Error during analysis: {str(e)}"
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=analyze_sales,
        inputs=gr.JSON(label="Retail Sales Data"),
        outputs=gr.Textbox(label="Analysis Results"),
        title="UK Retail Sales Analysis",
        description="Enter retail sales data to get analysis results.",
        examples=[
            [{"data": [{"value": 100}, {"value": 120}, {"value": 110}]}],
            [{"data": [{"value": 90}, {"value": 85}, {"value": 95}]}]
        ]
    )
    
    return interface

if __name__ == "__main__":
    interface = create_retail_analysis_interface()
    interface.launch() 