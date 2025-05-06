import asyncio
from dotenv import load_dotenv
from uk_economic_data.ui.dashboard import DashboardUI
from uk_economic_data.services.data_loader import DataLoader

async def main() -> None:
    # Load environment variables
    load_dotenv()
    
    # Initialize data loader
    data_loader = DataLoader()
    df = data_loader.load_data()
    if df is None:
        print("Failed to load data.")
        return

    # Create and launch dashboard
    dashboard = DashboardUI()
    interface = dashboard.build_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    asyncio.run(main()) 