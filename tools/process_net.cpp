#include "../src/nnue.h"

#include <cstring>
#include <fstream>

using namespace NNUE;

int main(int, char** argv) {
  Net* rawContent = new Net();
  Net* Weights = new Net();

  std::ifstream(argv[1], std::ios::binary).read((char*) rawContent, sizeof(Net));

  memcpy(Weights, rawContent, sizeof(Net));

  // dpbusd preprocessing:
  for (int bucket = 0; bucket < OutputBuckets; bucket++)
    for (int i = 0; i < L1; i += 4)
      for (int j = 0; j < L2; ++j)
        for (int k = 0; k < 4; k++)
          Weights->L1WeightsAlt[bucket][i * L2 + j * 4 + k] = rawContent->L1Weights[bucket][i + k][j];

  // Transpose weights so that we don't need to permute after packus, because
  // it interleaves each 128 block from a and each 128 block from b, alternately.
  // Instead we want it to concatenate a and b
  constexpr int weightsPerBlock = sizeof(__m128i) / sizeof(int16_t);
  constexpr int NumRegs = sizeof(VecI) / 8;
  __m128i regs[NumRegs];

  __m128i* ftWeights = (__m128i*) Weights->FeatureWeights;
  __m128i* ftBiases = (__m128i*) Weights->FeatureBiases;

  for (int i = 0; i < KingBuckets * 768 * L1 / weightsPerBlock; i += NumRegs) {
    for (int j = 0; j < NumRegs; j++)
      regs[j] = ftWeights[i + j];
    for (int j = 0; j < NumRegs; j++)
      ftWeights[i + j] = regs[PackusOrder[j]];
  }

  for (int i = 0; i < L1 / weightsPerBlock; i += NumRegs) {
    for (int j = 0; j < NumRegs; j++)
      regs[j] = ftBiases[i + j];
    for (int j = 0; j < NumRegs; j++)
      ftBiases[i + j] = regs[PackusOrder[j]];
  }

  std::ofstream(argv[2], std::ios::binary).write((char*) Weights, sizeof(Net));

  return 0;
}
