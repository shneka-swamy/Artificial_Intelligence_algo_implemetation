import torch

def determine_average(x):
    B, T, C = x.shape
    wei = torch.tril(torch.ones(T, T))
    wei = wei/ torch.sum(wei, 1, keepdim=True)
    xbow2 = wei @ x #(T, T) X (B, T, C) ---> (B, T, T) X (B, T, C) --> (B, T, C)
    print(torch.allclose(x, xbow2))


def main():
    torch.manual_seed(1337)
    B, T, C = 4, 8, 2
    x = torch.randn(B, T, C)
    determine_average(x)


if __name__ == '__main__':
    main()