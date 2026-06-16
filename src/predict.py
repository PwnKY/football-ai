from features import BASE_ODDS_FEATURES, build_single_match_features
from utils import MODELS_DIR, load_json, load_pickle


RESULT_NAMES = {
    0: "主胜",
    1: "平局",
    2: "客胜",
}


def ask_float(prompt):
    """
    Ask the user for a number.

    Empty input is allowed and will be treated as 0 if the trained model needs
    that feature. This keeps prediction simple for early experiments.
    """
    value = input(prompt).strip()
    if value == "":
        return None
    return float(value)


def main():
    model_path = MODELS_DIR / "football_model.pkl"
    features_path = MODELS_DIR / "features.json"

    model = load_pickle(model_path)
    feature_names = load_json(features_path)

    print("请输入一场比赛的赔率数据。没有的数据可以直接回车跳过。")

    match_data = {}
    for col in BASE_ODDS_FEATURES:
        match_data[col] = ask_float(f"{col}: ")

    # Ask handicap inputs only if the trained model used handicap_change.
    if "handicap_change" in feature_names:
        match_data["opening_handicap_line"] = ask_float("opening_handicap_line: ")
        match_data["closing_handicap_line"] = ask_float("closing_handicap_line: ")

    # Ask over/under inputs only if the trained model used over_under_change.
    if "over_under_change" in feature_names:
        match_data["opening_over_under_line"] = ask_float("opening_over_under_line: ")
        match_data["closing_over_under_line"] = ask_float("closing_over_under_line: ")

    X = build_single_match_features(match_data, feature_names)

    probabilities = model.predict_proba(X)[0]
    predicted_class = int(probabilities.argmax())

    print()
    print(f"主胜概率：{probabilities[0] * 100:.1f}%")
    print(f"平局概率：{probabilities[1] * 100:.1f}%")
    print(f"客胜概率：{probabilities[2] * 100:.1f}%")
    print(f"模型倾向：{RESULT_NAMES[predicted_class]}")


if __name__ == "__main__":
    main()
