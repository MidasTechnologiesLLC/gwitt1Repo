# main.py
import yaml
import logging
from pathlib import Path
from midas.data_processor import DataProcessor
from midas.feature_engineer import FeatureEngineer
from midas.hmm_trainer import HMMTrainer
from midas.analysis import MarketRegimeAnalysis

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("midas.log"),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Load config
        with open('config/params.yaml') as f:
            config = yaml.safe_load(f)
            
        # Create output directory
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process data
        processor = DataProcessor(config)
        raw_data = processor.process_tickers()
        
        # Engineer features
        engineer = FeatureEngineer(config)
        features = engineer.calculate_features(raw_data)
        
        # Train HMM
        trainer = HMMTrainer(config)
        model = trainer.train(features)
        trainer.save_model(output_dir / "hmm_model.pkl")
        
        # Generate analysis
        analysis = MarketRegimeAnalysis(model, features)
        
        # Save plots
        analysis.plot_regimes(pd.concat([d['close'] for d in raw_data.values]))
        plt.savefig(output_dir / "combined_regimes.png")
        
        analysis.plot_transition_matrix()
        plt.savefig(output_dir / "transition_matrix.png")
        
        analysis.plot_state_durations()
        plt.savefig(output_dir / "state_durations.png")
        
        logging.info("Process completed successfully")
        
    except Exception as e:
        logging.error(f"Main process failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
