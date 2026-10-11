"""Run B's label-free layout head and zero-initialized fusion projection."""

import math

import torch
from torch import nn


class LayoutAttentionBlock(nn.Module):
    """Cross attention to context, self attention, then a feed-forward block."""

    def __init__(self, width=256, heads=4):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, query, context):
        context_norm = self.context_norm(context)
        update, _ = self.cross_attention(
            self.query_norm(query), context_norm, context_norm,
            need_weights=False,
        )
        query = query + update
        query_norm = self.self_norm(query)
        update, _ = self.self_attention(
            query_norm, query_norm, query_norm, need_weights=False,
        )
        query = query + update
        return query + self.ffn(self.ffn_norm(query))


class DualViewHead(nn.Module):
    """Predict four instance slots from full-mask features, without GT inputs.

    Production dimensions are 28 x 28, encoder width 768, head width 256,
    four slots and 20 JT classes. Smaller grids/widths support CPU contracts.
    """

    def __init__(self, encoder_width=768, width=256, grid_size=28,
                 slots=4, classes=20, heads=4):
        super().__init__()
        self.width = width
        self.grid_size = grid_size
        self.slots = slots
        self.classes = classes
        self.query_projection = nn.Sequential(
            nn.LayerNorm(6 * encoder_width),
            nn.Linear(6 * encoder_width, width),
        )
        self.reference_projection = nn.Linear(2 * encoder_width, width)
        self.source_projection = nn.Linear(encoder_width, width)
        self.row_embedding = nn.Parameter(torch.empty(grid_size, width))
        self.column_embedding = nn.Parameter(torch.empty(grid_size, width))
        self.source_embedding = nn.Parameter(torch.empty(3, width))
        self.blocks = nn.ModuleList([
            LayoutAttentionBlock(width, heads) for _ in range(2)
        ])
        self.slot_queries = nn.Parameter(torch.empty(slots, width))
        self.slot_norm = nn.LayerNorm(width)
        self.token_norm = nn.LayerNorm(width)
        self.slot_attention = nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.slot_ffn_norm = nn.LayerNorm(width)
        self.slot_ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(),
            nn.Linear(4 * width, width),
        )
        self.existence = nn.Linear(width, 1)
        self.classifier = nn.Linear(width, classes)
        for parameter in (self.row_embedding, self.column_embedding,
                          self.source_embedding, self.slot_queries):
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(self, quadrants, full_features):
        x_top, y_top, x_bottom, y_bottom = quadrants
        batch, height, width, _ = x_bottom.shape
        if (height, width) != (self.grid_size, self.grid_size):
            raise ValueError("Dual-view head requires its configured square grid")
        if full_features.shape[1:3] != (height, width):
            raise ValueError("Full features must contain only the lower grid")
        position = (self.row_embedding[:, None]
                    + self.column_embedding[None, :])
        query = self.query_projection(torch.cat(
            (x_bottom, y_bottom, full_features), dim=-1
        )) + position + self.source_embedding[0]
        reference = self.reference_projection(torch.cat(
            (x_top, y_top), dim=-1
        )) + position + self.source_embedding[1]
        source = (self.source_projection(x_bottom)
                  + position + self.source_embedding[2])
        query = query.reshape(batch, height * width, self.width)
        context = torch.cat((reference.flatten(1, 2), source.flatten(1, 2)), 1)
        for block in self.blocks:
            query = block(query, context)

        slots = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        query_norm = self.token_norm(query)
        update, _ = self.slot_attention(
            self.slot_norm(slots), query_norm, query_norm, need_weights=False,
        )
        slots = slots + update
        slots = slots + self.slot_ffn(self.slot_ffn_norm(slots))
        occupancy = torch.einsum("bkd,bpd->bkp", slots, query)
        occupancy = occupancy / math.sqrt(self.width)
        existence = self.existence(slots).squeeze(-1)
        classes = self.classifier(slots)
        gates = occupancy.sigmoid() * existence.sigmoid().unsqueeze(-1)
        slot_information = torch.cat((slots, classes.softmax(-1)), dim=-1)
        field = torch.einsum("bkp,bkd->bpd", gates, slot_information)
        field = field / self.slots
        layout = torch.cat((field, query), dim=-1)
        return {
            "layout": layout.reshape(batch, height, width, -1),
            "occupancy_logits": occupancy.reshape(
                batch, self.slots, height, width
            ),
            "existence_logits": existence,
            "class_logits": classes,
            "slots": slots,
            "query": query,
        }


class DualViewFusion(nn.Module):
    """532 -> 3072 production residual; last weight and bias start at zero."""

    def __init__(self, encoder_width=768, width=256, classes=20):
        super().__init__()
        self.norm = nn.LayerNorm(2 * width + classes)
        self.projection = nn.Linear(2 * width + classes, 4 * encoder_width)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, layout, has_jt=None):
        # Route BEFORE the projection: BF still updates the fusion projection,
        # while its layout head receives zero gradients through this branch.
        if has_jt is not None:
            active = has_jt.to(device=layout.device, dtype=torch.bool)
            layout = torch.where(
                active[:, None, None, None], layout, layout.detach()
            )
        return self.projection(self.norm(layout))
