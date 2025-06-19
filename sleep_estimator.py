import pandas as pd
import torch


def read_health_data():
    health_csv = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv", index_col=0, header=0)

    # 読み込んだデータの確認
    # print(health_csv.head())

    # NNで処理できるようにカテゴリカルデータを数値に変換
    health_data = health_csv.copy()
    # カテゴリカルデータを数値に変換
    health_data["Gender"] = health_data["Gender"].map({"Male": 0, "Female": 1})
    health_data["BMI Category"] = health_data["BMI Category"].map(
        {"Normal": 0, "Normal Weight": 0, "Overweight": 1, "Obese": 2}
    )
    # Sleep Disorderのnan(None)を"None"の文字列に置き換え
    health_data["Sleep Disorder"] = health_data["Sleep Disorder"].fillna("None")
    # Sleep Disorderを数値に変換
    health_data["Sleep Disorder"] = health_data["Sleep Disorder"].map({"None": 0, "Sleep Apnea": 1, "Insomnia": 2})
    # 不要な列を削除
    health_data = health_data.drop("Occupation", axis=1)
    # 血圧の上下を分ける
    a = health_data["Blood Pressure"].str.partition("/")
    health_data["Blood High"] = a[0].astype(float)
    health_data["Blood Low"] = a[2].astype(float)
    # 分割前の血圧の列を削除
    health_data = health_data.drop("Blood Pressure", axis=1)

    # 変換後のデータの確認
    # print(health_data.head())

    return health_data


# データをPyTorchでの学習に利用できる形式に変換
def create_dataset_from_dataframe(data, target_tag="tip"):
    # "tip"の列を目的にする
    target = torch.tensor(data[target_tag].values, dtype=torch.float32).reshape(-1, 1)
    # "tip"以外の列を入力にする
    input = torch.tensor(data.drop(target_tag, axis=1).values, dtype=torch.float32)
    return input, target


# 4層順方向ニューラルネットワークモデルの定義
class FourLayerNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h1 = torch.tanh(self.l1(x))
        h2 = torch.tanh(self.l2(h1))
        o = self.l3(h2)
        return o


def train_model(nn_model, input, target):
    # データセットの作成
    tips_dataset = torch.utils.data.TensorDataset(input, target)
    # バッチサイズ=50として学習用データローダを作成
    train_loader = torch.utils.data.DataLoader(tips_dataset, batch_size=50, shuffle=True)

    # オプティマイザ
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=1e-14, momentum=0.9)

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
health_data = read_health_data()
input, target = create_dataset_from_dataframe(health_data, target_tag="Quality of Sleep")

# NNのオブジェクトを作成
nn_model = FourLayerNN(input.shape[1], 30, 1)
train_model(nn_model, input, target)

# 学習後のモデルの保存
# torch.save(nn_model.state_dict(), "nn_model.pth")

# 学習後のモデルのテスト
test_data = torch.tensor(
    [
        [
            0,  # 0:Male, 1:Female
            29,  # Age
            6.3,  # Sleep Duration
            40,  # Physical Activity
            7,  # Stress Level
            2,  # BMI 0:Normal 0:Normal Weight 1:Overweight 2:Obese
            140,  # Blood High
            90,  # Blood Low
            82,  # Heart Rate
            3500,  # Steps
            2,  # Sleep Disorder 0:None 1:Sleep Apnea 2:Insomnia
        ],
        [1, 34, 5.8, 32, 8, 1, 131, 86, 81, 5200, 1],
        [1, 52, 8.4, 30, 3, 0, 125, 80, 65, 5000, 0],
    ],
    dtype=torch.float32,
)
with torch.inference_mode():  # 推論モード（学習しない）
    print(nn_model(test_data))
