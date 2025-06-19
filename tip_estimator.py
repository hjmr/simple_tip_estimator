import pandas as pd
import torch


def read_data() -> pd.DataFrame:
    # データの読み込み
    tips_csv = pd.read_csv("tips.csv", index_col=None, header=0)
    # 読み込んだデータの確認
    # print(tips_csv.head())

    # NNで処理できるようにカテゴリカルデータを数値に変換
    tips_data = tips_csv.copy()
    tips_data["sex"] = tips_data["sex"].map({"Male": 0, "Female": 1})
    tips_data["smoker"] = tips_data["smoker"].map({"No": 0, "Yes": 1})
    tips_data["time"] = tips_data["time"].map({"Dinner": 0, "Lunch": 1})
    tips_data["day"] = tips_data["day"].map({"Sun": 0, "Sat": 1, "Thur": 2, "Fri": 3})

    # 数値の調整
    tips_data["total_bill"] = tips_data["total_bill"] / 10

    # 変換後のデータの確認
    # print(tips_data.head())

    return tips_data


# データをPyTorchでの学習に利用できる形式に変換
def create_dataset_from_dataframe(
    tips_data: pd.DataFrame, target_tag: str = "tip"
) -> tuple[torch.Tensor, torch.Tensor]:
    # "tip"の列を目的にする
    target = torch.tensor(tips_data[target_tag].values, dtype=torch.float32).reshape(-1, 1)
    # "tip"以外の列を入力にする
    input = torch.tensor(tips_data.drop(target_tag, axis=1).values, dtype=torch.float32)
    return input, target


# 4層順方向ニューラルネットワークモデルの定義
class FourLayerNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(self.l1(x))
        h2 = torch.tanh(self.l2(h1))
        o = self.l3(h2)
        return o


def train_model(nn_model: FourLayerNN, input: torch.Tensor, target: torch.Tensor) -> None:
    # データセットの作成
    tips_dataset = torch.utils.data.TensorDataset(input, target)
    # バッチサイズ=25として学習用データローダを作成
    train_loader = torch.utils.data.DataLoader(tips_dataset, batch_size=25, shuffle=True)

    # オプティマイザ
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.01, momentum=0.9)

    # データセット全体に対して10000回学習
    for epoch in range(10000):
        # バッチごとに学習する
        for x, y_hat in train_loader:
            y = nn_model(x)
            loss = torch.nn.functional.mse_loss(y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 1000回に1回テストして誤差を表示
        if epoch % 1000 == 0:
            with torch.inference_mode():  # 推論モード（学習しない）
                y = nn_model(input)
                loss = torch.nn.functional.mse_loss(y, target)
                print(epoch, loss)


# データの準備
tips_data = read_data()
input, target = create_dataset_from_dataframe(tips_data)

# NNのオブジェクトを作成
nn_model = FourLayerNN(input.shape[1], 30, 1)
train_model(nn_model, input, target)

# 学習後のモデルの保存
# torch.save(nn_model.state_dict(), "nn_model.pth")

# 学習後のモデルのテスト
test_data = torch.tensor(
    [
        [
            3.0,  # total_bill
            1,  # 0:Male 1:Female
            1,  # Smoke 0:No 1:Yes
            1,  # Day 0:Sun 1:Sat 2:Thur 3:Fri
            0,  # Time 0:Dinner 1:Lunch
            2,  # # of Members
        ]
    ],
    dtype=torch.float32,
)
with torch.inference_mode():  # 推論モード（学習しない）
    print(nn_model(test_data))
