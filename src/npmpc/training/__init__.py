import logging
import time

import torch

from npmpc.nps.NeuralProcess import NeuralProcess

logger = logging.getLogger(__name__)


class Training:
    
    optimizer = None
    gaussian_kl_factor: float = 1.0
    model: NeuralProcess
    
    def __init__(self, model: NeuralProcess):
        self.model = model
        
    
    def train(self, train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset, params: dict):
        self.params = params

        sched = params['scheduler']
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=sched['learning_rate'], weight_decay=1e-2)
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], num_workers=params['num_workers'], shuffle=True, pin_memory=True)
        # Reduce the LR when the validation loss plateaus.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            factor=sched['factor'],
            patience=sched['patience'],
            cooldown=5,
            threshold=1e-3,
        )

        torch.manual_seed(params['seed'])

        self.model.send_to_device(params['device'])

        logger.info(f"Starting training: {params['num_epochs']} epochs, batch_size={params['batch_size']}, device={params['device']}")
        t_start = time.time()

        for epoch in range(params['num_epochs']+1):
            epoch_loss = 0; epoch_R2 = 0
            counter = 0
            for data in data_loader:
                counter += 1
                data['x'] = data['x'].to(params['device'], non_blocking=True)
                data['y'] = data['y'].to(params['device'], non_blocking=True)
                loss, R2 = self.iteration_step(epoch, data)
                epoch_loss += loss; epoch_R2 += R2

            if epoch % params['test_interval'] == 0:
                val_loss, val_r2 = self.validate(test_dataset, params)
                scheduler.step(val_loss)
                logger.info(
                    f"Epoch {epoch:4d}/{params['num_epochs']} | "
                    f"train_loss={epoch_loss / counter:+.4f}  train_R2={epoch_R2 / counter:5.2f}%  "
                    f"val_loss={val_loss:+.4f}  val_R2={val_r2:5.2f}%"
                )
            if epoch % params['save_interval'] == 0:
                self.model.save_model(epoch)

        self.model.save_model('final')
        logger.info(f"Training finished in {time.time() - t_start:.1f}s")


    def iteration_step(self, step: int, data: dict):
        min_context_size = self.params['train_min_context_size']
        max_context_size = min(data['x'].shape[-2], self.params['train_max_context_size'])
        context_size = torch.randint(min_context_size, max_context_size, (data['x'].shape[-3],)).to(data['x'].device)
        context_mask = torch.arange(max_context_size).unsqueeze(0).to(data['x'].device) < context_size.unsqueeze(-1) 
        
        # To ensure the subset of context points is always different 
        permed_ids = torch.randperm(data['x'].shape[-2])
        x = data['x'][...,permed_ids,:]; y = data['y'][...,permed_ids,:]

        z = self.model.encode(x[...,:max_context_size,:], y[...,:max_context_size,:], context_mask)

        # Decode the prediction
        y_mu, y_sigma = self.model.decode(x, z=z)

        # Compute the loss (decoder-specific)
        output_loss = self.model.decoder.loss(y_mu, y_sigma, y)
        mean_loss = output_loss.mean()

        # Backpropagate
        mean_loss.backward()

        if not self.model.check_nans():
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Other Metrics computation
        y_R2 = 100*(1-(y_mu-y).pow(2).sum(-1).mean() / (y-y.mean()).pow(2).sum(-1).mean())

        return mean_loss.item(), y_R2.item()
        
        
    def validate(self, dataset: torch.utils.data.Dataset, params: dict):
        self.model.send_to_device(params['device'])
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=params['test_batch_size'], num_workers=params['num_workers'], shuffle=False, pin_memory=True)
        
        epoch_loss = 0; epoch_R2 = 0

        with torch.no_grad():
            counter = 0
            for data in data_loader:
                counter += 1
                data['x'] = data['x'].to(params['device'], non_blocking=True)
                data['y'] = data['y'].to(params['device'], non_blocking=True)

                context_size = torch.tensor([params['test_context_size']]*data['x'].shape[0]).to(data['x'].device)
                context_mask = torch.arange(context_size.max()).unsqueeze(0).to(data['x'].device) < context_size.unsqueeze(-1) 
                
                # To ensure the subset of context points is always different 
                permed_ids = torch.randperm(data['x'].shape[-2])
                x = data['x'][...,permed_ids,:]; y = data['y'][...,permed_ids,:]
                
                z = self.model.encode(x[...,:params['test_context_size'],:], y[...,:params['test_context_size'],:], context_mask)

                # Decode the prediction
                y_mu, y_sigma = self.model.decode(x, z=z)

                # Compute metrics (decoder-specific loss)
                output_loss = self.model.decoder.loss(y_mu, y_sigma, y)
                epoch_loss += output_loss.mean()
                epoch_R2 += 100*(1-(y_mu-y).pow(2).sum(-1).mean() / (y-y.mean()).pow(2).sum(-1).mean())

        return (epoch_loss / counter).item(), (epoch_R2 / counter).item()

from npmpc.training.furuta import FurutaTraining

TRAINING_REGISTRY = {
    'furuta': FurutaTraining,
}
