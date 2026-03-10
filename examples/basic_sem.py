import pandas as pd

from psysem import SEMModel


def main() -> None:
    data = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [1.2, 1.9, 3.2, 3.9],
            "y": [0.9, 2.1, 2.8, 4.2],
        }
    )
    model = SEMModel("y ~ x1 + x2")
    result = model.fit(data)
    print(result.summary())


if __name__ == "__main__":
    main()
