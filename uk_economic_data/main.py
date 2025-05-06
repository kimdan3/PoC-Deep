import asyncio
from dotenv import load_dotenv
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

if __name__ == "__main__":
    asyncio.run(main()) 