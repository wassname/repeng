from repeng.train.train_svft import main, TrainingConfig
import tyro 

if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)
    main(config)
