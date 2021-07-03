import torch
import numpy as np
from itertools import groupby
import pdb
import logging

def dynamic_matching(
    tensor1, tensor2, prob1=None, prob2=None, 
):

    tensor1 = tensor1.tolist()
    tensor2 = tensor2.tolist()
    prob1 = prob1.tolist()
    prob2 = prob2.tolist()
    M, N = len(tensor1), len(tensor2)

    dp = [[0 for _ in range(N + 1)] for _ in range(M + 1)]
    dp[0][0] = 0, [], []
    s1, s2 = 0, 0
    if len(tensor1) > 0:
        s1 = tensor1[0]

    if len(tensor2) > 0:
        s2 = tensor2[0]

    for i in range(1, N + 1):
        dp[0][i] = (
            i,
            dp[0][i - 1][1] + [(-1, tensor2[i - 1])],
            dp[0][i - 1][2] + [(0, prob2[i - 1])],
        )
    for i in range(1, M + 1):
        dp[i][0] = (
            i,
            dp[i - 1][0][1] + [(tensor1[i - 1], -1)],
            dp[i - 1][0][2] + [(prob1[i - 1], 0)],
        )

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if tensor1[i - 1] == tensor2[j - 1]:
                dp[i][j] = (
                    dp[i - 1][j - 1][0],
                    dp[i - 1][j - 1][1] + [(tensor1[i - 1], tensor2[j - 1])],
                    dp[i - 1][j - 1][2] + [(prob1[i - 1], prob2[j - 1])],
                )
            else:
                num, idx = torch.min(
                    torch.tensor(
                        [dp[i - 1][j - 1][0], dp[i - 1][j][0], dp[i][j - 1][0]]
                    ),
                    0,
                )
                dp[i][j] = [0, 0, 0]
                dp[i][j][0] = 1 + num
                if idx == 0:
                    dp[i][j][1] = dp[i - 1][j - 1][1] + [
                        (tensor1[i - 1], tensor2[j - 1])
                    ]
                    dp[i][j][2] = dp[i - 1][j - 1][2] + [(prob1[i - 1], prob2[j - 1])]
                if idx == 1:
                    dp[i][j][1] = dp[i - 1][j][1] + [
                        (tensor1[i - 1], -1)
                    ]
                    dp[i][j][2] = dp[i - 1][j][2] + [(prob1[i - 1], 0)]
                if idx == 2:
                    dp[i][j][1] = dp[i][j - 1][1] + [
                        (-1, tensor2[j - 1])
                    ]
                    #dp[i][j][1] = dp[i][j - 1][1]
                    dp[i][j][2] = dp[i][j - 1][2] + [(0, prob2[j - 1])]
                    #dp[i][j][2] = dp[i][j - 1][2]

    return dp[-1][-1][1], dp[-1][-1][2]
