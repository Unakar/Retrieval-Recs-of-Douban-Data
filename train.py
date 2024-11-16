import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import pandas as pd
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from dataloader import Rating_Dataloader, ID_Mapper, Contact_Dataloader, collate_fn
from models.MatrixFactor import MatrixFactorization
from models.graphrec import GraphRec

# 按用户分组计算NDCG
def compute_ndcg(group):
    true_ratings = group['true'].tolist()
    pred_ratings = group['pred'].tolist()
    return ndcg_score([true_ratings], [pred_ratings], k = 50)

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader = yaml.FullLoader)

class RecommendationModel(LightningModule):
    def __init__(self, type: MediaType):
        super().__init__()
        self.type = type
        self.tag_embedding = TagEmbedding()
        self.configure_paths()
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"

        self.train_loader, self.test_loader = self._init_dataset()
        self.model = Model(
            len(self.user_idx_converter), len(self.item_idx_converter)
        ).to(self.device)

        if Config.PRETRAINED:
            self.model.load_state_dict(torch.load(self.pretrained_path, map_location=self.device))

        self.loss_fn = nn.MSELoss()

    def configure_paths(self):
        if self.type == MediaType.BOOK:
            self.score_data_path = Config.BOOK_SCORE_PATH
            self.type_name = "Book"
        else:
            self.score_data_path = Config.MOVIE_SCORE_PATH
            self.type_name = "Movie"
        # Define paths dynamically for SAVE_MODEL_PATH, PRETRAINED_PATH, RESULT_PATH, etc.

    def _init_dataset(self):
        self.tag_embedding_dict = self.tag_embedding.get_embedding(self.type)
        loaded_data = pd.read_csv(self.score_data_path)
        self.user_idx_converter = IdxConverter(loaded_data["User"].unique())
        self.item_idx_converter = IdxConverter(loaded_data[self.type_name].unique())

        train_data, test_data = train_test_split(
            loaded_data, test_size=0.5, random_state=42
        )

        train_dataset = RatingDataset(
            self.type,
            train_data,
            self.user_idx_converter,
            self.item_idx_converter,
            self.tag_embedding_dict,
        )
        test_dataset = RatingDataset(
            self.type,
            test_data,
            self.user_idx_converter,
            self.item_idx_converter,
            self.tag_embedding_dict,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, drop_last=True
        )

        return train_loader, test_loader

    def forward(self, user_ids, item_ids, tag_embedding):
        return self.model(user_ids, item_ids, tag_embedding)

    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, tag_embedding = batch
        predictions = self.forward(user_ids, item_ids, tag_embedding.squeeze(1))
        loss = self.loss_fn(predictions, ratings)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, true_ratings, tag_embedding = batch
        predictions = self.forward(user_ids, item_ids, tag_embedding.squeeze(1))
        loss = self.loss_fn(predictions, true_ratings)
        pred_ratings = self.model.prediction_to_rating(predictions)
        return {'loss': loss, 'pred': pred_ratings, 'true': true_ratings, 'user_ids': user_ids}

    def validation_epoch_end(self, outputs):
        total_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        results = []
        for output in outputs:
            user_ids_np = output['user_ids'].long().cpu().numpy().reshape(-1, 1)
            pred_ratings_np = output['pred'].cpu().numpy().reshape(-1, 1)
            true_ratings_np = output['true'].numpy().reshape(-1, 1)
            batch_results = np.column_stack((user_ids_np, pred_ratings_np, true_ratings_np))
            results.append(batch_results)

        results = np.vstack(results)
        results_df = pd.DataFrame(results, columns=["user", "pred", "true"])
        results_df["user"] = results_df["user"].astype(int)
        results_df.to_csv(self.result_path)

        ndcg_scores = results_df.groupby("user").apply(compute_ndcg)
        ndcg_scores = ndcg_scores.loc[ndcg_scores != 0]

        avg_ndcg = ndcg_scores.mean()
        self.log('avg_ndcg', avg_ndcg)
        self.log('val_loss', total_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=Config.STEP_LR_STEP_SIZE,
            gamma=Config.STEP_LR_GAMMA,
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.test_loader


# Use a Trainer to run the training loop
if __name__ == "__main__":
    type = MediaType.BOOK  # or MediaType.MOVIE
    model = RecommendationModel(type)

    trainer = Trainer(
        max_epochs=Config.NUM_EPOCHS,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',
                dirpath='./checkpoints',
                filename='best-checkpoint',
                save_top_k=1,
                mode='min'
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
    )