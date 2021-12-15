from pytorch_tabnet.tab_network import TabNet as _TabNet


class TabNet(_TabNet):
    def forward(self, x):
        return super().forward(x)[0]
