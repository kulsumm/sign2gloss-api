import torch
import torch.nn as nn

class Sign2Gloss(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, emb_dim=256, num_layers=2):
        super().__init__()

        # Encoder: BiLSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Project BiLSTM (2H) → H
        self.ctx_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Embedding + dropout
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.3)

        # Decoder: GRU
        self.decoder = nn.GRU(
            input_size=emb_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.3
        )

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, keypoints, gloss_ids):
        enc_out, _ = self.encoder(keypoints)
        ctx = enc_out.mean(dim=1)
        ctx = self.ctx_proj(ctx)
        emb = self.emb_dropout(self.emb(gloss_ids))
        ctx_rep = ctx.unsqueeze(1).repeat(1, emb.size(1), 1)
        dec_in = torch.cat([emb, ctx_rep], dim=-1)
        dec_out, _ = self.decoder(dec_in)
        logits = self.fc_out(dec_out)
        return logits

    def generate(self, kp, stoi, max_len=20):
        """
        kp: tensor [1, T, 99]
        """
        self.eval()
        with torch.no_grad():
            enc_out, _ = self.encoder(kp)
            ctx = self.ctx_proj(enc_out.mean(dim=1))
            hidden = ctx.unsqueeze(0)  # [1, 1, H]

            inp = torch.tensor([[stoi["<bos>"]]], device=kp.device)
            outputs = []

            for _ in range(max_len):
                emb = self.emb(inp)
                dec_in = torch.cat([emb, ctx.unsqueeze(1)], dim=-1)
                out, hidden = self.decoder(dec_in, hidden)
                logits = self.fc_out(out[:, -1, :])
                next_id = logits.argmax(dim=-1).item()

                if next_id == stoi["<eos>"]:
                    break

                outputs.append(next_id)
                inp = torch.tensor([[next_id]], device=kp.device)

        return outputs
