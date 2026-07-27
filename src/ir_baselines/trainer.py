import tqdm


class Trainer:
    """
    One training loop for both families.

    The criterion is chosen from model.LOSS by train.py; this class only needs
    to know how to feed the model and how to cast the label, both of which
    follow from LOSS.
    """

    def __init__(self, model, optimizer, criterion, scheduler, metric, data_loader, device):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler
        self._metric = metric
        self._data_loader = data_loader
        self._device = device

    def _forward(self, batch):
        if self._model.ENCODING == 'pair':
            score, _ = self._model(
                batch['input_ids'].to(self._device),
                batch['attention_mask'].to(self._device),
                batch['token_type_ids'].to(self._device),
            )
        else:
            score = self._model(
                batch['query_input_ids'].to(self._device),
                batch['query_attention_mask'].to(self._device),
                batch['doc_input_ids'].to(self._device),
                batch['doc_attention_mask'].to(self._device),
            )

        label = batch['label'].to(self._device)
        if self._model.LOSS == 'cross_entropy':
            label = label.long()
        else:
            label = label.float()
        return score, label

    def make_train_step(self):
        def train_step(train_batch):
            self._model.train()
            self._optimizer.zero_grad()
            batch_score, label = self._forward(train_batch)
            batch_loss = self._criterion(batch_score, label)
            batch_loss.backward()
            self._optimizer.step()
            self._scheduler.step()
            return batch_loss.item()

        return train_step

    def train(self):
        train_step = self.make_train_step()
        epoch_loss = 0
        num_batch = len(self._data_loader)
        for _, batch in tqdm.tqdm(enumerate(self._data_loader), total=num_batch):
            epoch_loss += train_step(batch)
        return epoch_loss
