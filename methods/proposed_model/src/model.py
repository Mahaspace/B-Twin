import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperparamters import (
    ALPHA,
    BETA,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    EPSILON,
    GAMMA,
    HIDDEN_SIZE,
    PROPENSITY_HIDDEN_DIMS,
)
from .loss import HuberLoss


class Encoder(nn.Module):
    def __init__(
        self,
        n_features,
        seq_len,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
    ):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        input_dim = n_features * seq_len
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2_mean = nn.Linear(hidden_size, embedding_dim)
        self.fc2_logvar = nn.Linear(hidden_size, embedding_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return z, mean, logvar


class Decoder(nn.Module):
    def __init__(
        self, n_features, seq_len, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE
    ):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        input_dim = n_features * seq_len

        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_dim)

    def forward(self, z):
        z = F.relu(self.bn1(self.fc1(z)))
        z = self.fc2(z)
        z = z.view(z.size(0), self.seq_len, self.n_features)
        return z


class PropensityScoreEstimator(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden_dims=PROPENSITY_HIDDEN_DIMS):
        super(PropensityScoreEstimator, self).__init__()
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.propensity_score_estimator = nn.Sequential(*layers)

    def forward(self, z):
        z = self.propensity_score_estimator(z)
        p = torch.sigmoid(z)

        return p


class CausalInferenceModel(nn.Module):
    def __init__(
        self,
        n_features,
        seq_len,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        emb_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_SIZE,
        propensity_hidden_dims=PROPENSITY_HIDDEN_DIMS,
        epsilon=EPSILON,
    ):
        super().__init__()
        self.encoder = Encoder(
            n_features, seq_len, embedding_dim=emb_dim, hidden_size=hidden_dim
        )
        self.decoder = Decoder(
            n_features, seq_len, embedding_dim=emb_dim, hidden_size=hidden_dim
        )
        self.propensity_estimator = PropensityScoreEstimator(
            embedding_dim=emb_dim, hidden_dims=propensity_hidden_dims
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.emb_dim = emb_dim
        self.epsilon = epsilon
        self.hidden_dim = hidden_dim

    def forward(self, x):
        z, mean, logvar = self.encoder(x)
        x_reconstructed = self.decoder(z)
        propensity_score = self.propensity_estimator(z)
        ## propensity should be different than 0 or 1
        propensity_score = torch.clamp(propensity_score, self.epsilon, 1 - self.epsilon)
        return z, x_reconstructed, propensity_score, mean, logvar

    def loss_function(
        self, x, x_reconstructed, z, propensity_score, treatment, mean, logvar
    ):
        huber_loss = HuberLoss(delta=1.0)  # Initialize Huber Loss
        reconstruction_loss = huber_loss(x_reconstructed, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        propensity_loss = F.binary_cross_entropy(propensity_score.squeeze(), treatment)
        return (
            self.alpha * reconstruction_loss
            + self.beta * kl_loss
            + self.gamma * propensity_loss
        )


class CausalInferenceModelWithoutPropensity(nn.Module):
    def __init__(
        self,
        n_features,
        seq_len,
        emb_dim=EMBEDDING_DIM,
    ):
        super().__init__()
        self.encoder = Encoder(n_features, seq_len, embedding_dim=emb_dim)
        self.decoder = Decoder(n_features, seq_len, embedding_dim=emb_dim)

    def forward(self, x):
        z, mean, logvar = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed, mean, logvar

    def loss_function(self, x, x_reconstructed, mean, logvar):
        huber_loss = HuberLoss(delta=1.0)  # Initialize Huber Loss
        reconstruction_loss = huber_loss(x_reconstructed, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return ALPHA * reconstruction_loss + BETA * kl_loss


class WeightPredictorNN(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim_1=128, hidden_dim_2=64, dropout_rate=0
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 2, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, treated_latent, control_latent):
        n_treated, n_control = treated_latent.shape[0], control_latent.shape[0]

        treated_expanded = treated_latent.unsqueeze(1).repeat(1, n_control, 1)
        control_expanded = control_latent.unsqueeze(0).repeat(n_treated, 1, 1)
        combined_input = torch.cat((treated_expanded, control_expanded), dim=2)
        combined_input = combined_input.view(-1, combined_input.size(2))
        weights = self.fc(combined_input)

        weights = weights.view(n_treated, n_control)
        weights = torch.exp(weights)
        return weights


class NormalWeightPredictorNN(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim_1=128, hidden_dim_2=64, dropout_rate=0
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, treated_latent, control_latent):
        n_treated, n_control = treated_latent.shape[0], control_latent.shape[0]

        treated_expanded = treated_latent.unsqueeze(1).repeat(1, n_control, 1)
        control_expanded = control_latent.unsqueeze(0).repeat(n_treated, 1, 1)
        combined_input = torch.cat((treated_expanded, control_expanded), dim=2)
        combined_input = combined_input.view(-1, combined_input.size(2))
        weights = self.fc(combined_input)

        weights = weights.view(n_treated, n_control)
        weights = torch.softmax(weights, dim=1)

        return weights
