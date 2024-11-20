import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import pandas as pd
import numpy as np
import pytorch_lightning as pl
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

class Config: #由yaml赋予属性值即可，此处隐去服务器实际路径
    pass
    
class RecommenderDataModule(pl.LightningDataModule):
    def __init__(self, douban_type: DOUBAN_TYPE, batch_size: int):
        super().__init__()
        self.type = douban_type
        self.batch_size = batch_size
        self.tag_embedding = TagEmbedding()

        if self.type == DOUBAN_TYPE.BOOK:
            self.score_data_path = Config.BOOK_SCORE_PATH
            self.type_name = "Book"
        else:
            self.score_data_path = Config.MOVIE_SCORE_PATH
            self.type_name = "Movie"

        self.user_idx_converter = None
        self.item_idx_converter = None

    def setup(self, stage=None):
        # Load data and prepare datasets
        self.tag_embedding_dict = self.tag_embedding.get_embedding(self.type)
        loaded_data = pd.read_csv(self.score_data_path)
        self.user_idx = ID_Mapper(loaded_data["User"].unique())
        self.item_idx = ID_Mapper(loaded_data[self.type_name].unique())

        train_data, test_data = train_test_split(
            loaded_data, test_size=0.5, random_state=42
        )

        self.train_dataset = Rating_Dataloader(
            self.type,
            train_data,
            self.user_idx_converter,
            self.item_idx_converter,
            self.tag_embedding_dict,
        )
        self.test_dataset = Rating_Dataloader(
            self.type,
            test_data,
            self.user_idx_converter,
            self.item_idx_converter,
            self.tag_embedding_dict,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )


class RecommenderLightningModule(pl.LightningModule):
    def __init__(self, douban_type: DOUBAN_TYPE, num_users: int, num_items: int):
        super().__init__()
        self.type = douban_type
        self.model = Model(num_users, num_items)

        if self.type == "BOOK":
            self.save_model_path = Config.BOOK_SAVE_MODEL_PATH
            self.result_path = Config.BOOK_RESULT_PATH
        else:
            self.save_model_path = Config.MOVIE_SAVE_MODEL_PATH
            self.result_path = Config.MOVIE_RESULT_PATH


        self.loss_fn = nn.MSELoss()
        self.learning_rate = Config.LEARNING_RATE
        self.t_max = Config.COSINE_ANNEALING_T_MAX
        self.self.eta_min = Config.ETA_MIN

        self.avg_ndcg = None

    def forward(self, user_ids, item_ids, tag_embedding):
        return self.model(user_ids, item_ids, tag_embedding)

    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings, tag_embedding = batch
        predictions = self.model(
            user_ids,
            item_ids,
            tag_embedding.squeeze(1),
        )
        loss = self.loss_fn(predictions, ratings)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, true_ratings, tag_embedding = batch
        predictions = self.model(
            user_ids,
            item_ids,
            tag_embedding.squeeze(1),
        )
        loss = self.loss_fn(predictions, true_ratings)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        pred_ratings = self.model.prediction_to_rating(predictions)
        return {
            'user_ids': user_ids.cpu(),
            'pred_ratings': pred_ratings.cpu(),
            'true_ratings': true_ratings.cpu()
        }

    def validation_epoch_end(self, outputs):
        # Gather results from all validation steps
        user_ids_all = []
        pred_ratings_all = []
        true_ratings_all = []
        for output in outputs:
            user_ids_all.append(output['user_ids'])
            pred_ratings_all.append(output['pred_ratings'])
            true_ratings_all.append(output['true_ratings'])

        user_ids_all = torch.cat(user_ids_all).numpy()
        pred_ratings_all = torch.cat(pred_ratings_all).numpy()
        true_ratings_all = torch.cat(true_ratings_all).numpy()

        results = np.column_stack(
            (user_ids_all.reshape(-1, 1), pred_ratings_all.reshape(-1, 1), true_ratings_all.reshape(-1, 1))
        )
        results_df = pd.DataFrame(results, columns=['user', 'pred', 'true'])
        results_df['user'] = results_df['user'].astype(int)
        results_df.to_csv(self.result_path, index=False)

        ndcg_scores = results_df.groupby("user").apply(compute_ndcg)
        ndcg_scores = ndcg_scores.loc[ndcg_scores != 0]
        avg_ndcg = ndcg_scores.mean()
        self.avg_ndcg = avg_ndcg
        self.log('avg_ndcg', avg_ndcg, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.t_max,
            eta_min=self.eta_min,
            verbose=True,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def on_train_epoch_end(self):
        # Save the model every 5 epochs
        if (self.current_epoch + 1) % 5 == 0:
            torch.save(self.model.state_dict(), self.save_model_path)

    def on_train_end(self):
        # Save the final model
        torch.save(self.model.state_dict(), self.save_model_path_final)


def main():
    # Choose the media type: DOUBAN_TYPE.BOOK or DOUBAN_TYPE.MOVIE
    douban_type = Config.DOUBAN_TYPE

    # Initialize the data module
    data_module = RecommenderDataModule(
        douban_type=douban_type,
        batch_size=Config.BATCH_SIZE
    )
    data_module.setup()
    num_users = len(data_module.user_idx_converter)
    num_items = len(data_module.item_idx_converter)

    model = RecommenderLightningModule(
        douban_type=douban_type,
        num_users=num_users,
        num_items=num_items
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints/',
        filename='model-{epoch:02d}',
        save_top_k=-1,  # Save a checkpoint every time validation improves
        every_n_epochs=5,
    )

    trainer = pl.Trainer(
        max_epochs=Config.EPOCHS,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)
