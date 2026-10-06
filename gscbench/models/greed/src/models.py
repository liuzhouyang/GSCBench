import torch
import torch_geometric as tg


class EmbedModel(torch.nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, conv="gin", pool="add"):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.pre = torch.nn.Linear(self.input_dim, self.hidden_dim)

        if conv == "gin":
            make_conv = lambda: tg.nn.GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                )
            )
        elif conv == "gcn":
            make_conv = lambda: tg.nn.GCNConv(self.hidden_dim, self.hidden_dim)
        elif conv == "sage":
            make_conv = lambda: tg.nn.SAGEConv(self.hidden_dim, self.hidden_dim)
        elif conv == "gat":
            make_conv = lambda: tg.nn.GATConv(self.hidden_dim, self.hidden_dim)
        else:
            raise ValueError(f"Unsupported GREED conv '{conv}'.")

        self.convs = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.convs.append(make_conv())

        pooled_dim = self.hidden_dim * (self.n_layers + 1)
        if pool == "set":
            post_input_dim = pooled_dim * 2
        else:
            post_input_dim = pooled_dim
        self.post = torch.nn.Sequential(
            torch.nn.Linear(post_input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
        )

        if pool == "add":
            self.pool = tg.nn.global_add_pool
        elif pool == "mean":
            self.pool = tg.nn.global_mean_pool
        elif pool == "max":
            self.pool = tg.nn.global_max_pool
        elif pool == "sort":
            self.pool = tg.nn.global_sort_pool
        elif pool == "att":
            self.pool = tg.nn.GlobalAttention(
                torch.nn.Sequential(
                    torch.nn.Linear(pooled_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dim, 1),
                )
            )
        elif pool == "set":
            self.pool = tg.nn.Set2Set(pooled_dim, 1)
        else:
            raise ValueError(f"Unsupported GREED pool '{pool}'.")
        self.pool_str = pool

    def forward(self, g):
        x = g.x
        edge_index = g.edge_index

        x = self.pre(x)
        emb = x
        xres = x
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            if i & 1:
                x = x + xres
                xres = x
            x = torch.nn.functional.relu(x)
            emb = torch.cat((emb, x), dim=1)

        x = emb
        if self.pool_str == "sort":
            x = self.pool(x, g.batch, k=1)
        else:
            x = self.pool(x, g.batch)
        x = self.post(x)
        return x


class NormGEDModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_model = EmbedModel(
            args["n_layers"],
            args["input_dim"],
            args["hidden_dim"],
            args["output_dim"],
            conv=args["conv"],
            pool=args["pool"],
        )
        self.weighted = False
        self.lin = torch.nn.Linear(1, 1)

    def forward_emb(self, gx, hx):
        return torch.norm(gx - hx, dim=-1)

    def forward(self, g, h):
        gx = self.embed_model(g)
        hx = self.embed_model(h)
        return self.forward_emb(gx, hx)

