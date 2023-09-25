//-----------------------------------------------------------------------------
// Copyright (c) 2023 Qualcomm Innovation Center, Inc. All rights reserved. <br>
// SPDX-License-Identifier: BSD-3-Clause-Clear
//-----------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include "QAicApi.hpp"

namespace cloudai {

using VocabularyTableType = std::unordered_map<std::string, int64_t>;

class VocabularyHelper {
public:
  explicit VocabularyHelper(const std::string &pathToVocabFile)
      : pathToVocabularyFile_(pathToVocabFile) {
    if (!qaic_fs::exists(pathToVocabFile)) {
      throw std::runtime_error("Unable to find the vocabulary file : " +
                               pathToVocabFile);
    }
    populateVocabularyInfo();
  }

  [[nodiscard]] const VocabularyTableType &getVocabularyTable() const {
    return vocab_;
  }

  [[nodiscard]] const std::string &getVocabbularyFilePath() const {
    return pathToVocabularyFile_;
  }

  [[nodiscard]] const std::string &getWordAtIndex(int64_t index) const {
    if (index < wordVec_.size()) {
      return wordVec_[index];
    }

    return wordVec_[findIndexOfWord("[UNK]")];
  }

  [[nodiscard]] int64_t findIndexOfWord(const std::string &key) const {
    auto wordIter = vocab_.find(key);
    if (wordIter != vocab_.end()) {
      return wordIter->second;
    }

    auto unknownWordIter = vocab_.find("[UNK]");
    return unknownWordIter->second;
  }

private:
  VocabularyTableType vocab_;
  std::vector<std::string> wordVec_;
  std::string pathToVocabularyFile_;

  void populateVocabularyInfo() {
    std::ifstream file(pathToVocabularyFile_);
    if (!file.good()) {
      throw std::runtime_error("Unable to read the vocabulary file : " +
                               pathToVocabularyFile_);
    }

    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
      vocab_[line] = index;
      wordVec_.push_back(line);
      index++;
    }
  }
};

using shVocabularyHelper = std::shared_ptr<VocabularyHelper>;

class Tokenizer {
public:
  explicit Tokenizer(shVocabularyHelper vocabHelper)
      : vocabHelper_{vocabHelper} {}

  [[nodiscard]] std::vector<int64_t>
  getTokensForSentence(const std::string &sentence) {
    std::vector<int64_t> tokens;
    std::istringstream iss(sentence);
    std::string word;
    while (iss >> word) {
      auto indexOfWord = vocabHelper_->findIndexOfWord(word);
      tokens.push_back(indexOfWord);
    }
    return tokens;
  }

private:
  shVocabularyHelper vocabHelper_;
};

[[nodiscard]] std::string to_string(const qaic::rt::BufferMapping &bufMap) {
  std::ostringstream strm;
  strm << "Name : " << std::setw(20) << bufMap.bufferName << ", ";

  strm << "Type : " << std::setw(6);
  switch (bufMap.ioType) {
  case BUFFER_IO_TYPE_INVALID:
    strm << "INVALID";
    break;

  case BUFFER_IO_TYPE_INPUT:
    strm << "INPUT";
    break;

  case BUFFER_IO_TYPE_OUTPUT:
    strm << "OUTPUT";
    break;

  case BUFFER_IO_TYPE_INVAL:
    strm << "OUTPUT";
    break;
  }

  strm << ", Size = " << std::setw(8) << bufMap.size;

  strm << ", Data Type : " << std::setw(12);
  switch (bufMap.dataType) {
  case BUFFER_DATA_TYPE_FLOAT:
    strm << "32 bit FLOAT" << std::endl;
    break;

  case BUFFER_DATA_TYPE_FLOAT16:
    strm << "16 bit FLOAT" << std::endl;
    break;

  case BUFFER_DATA_TYPE_INT8Q:
    strm << "8 bit Quantized INT";
    break;

  case BUFFER_DATA_TYPE_UINT8Q:
    strm << "8 bit Quantized UNSIGNED INT";
    break;

  case BUFFER_DATA_TYPE_INT16Q:
    strm << "16 bit Quantized INT";
    break;

  case BUFFER_DATA_TYPE_INT32Q:
    strm << "32 bit Quantized INT";
    break;

  case BUFFER_DATA_TYPE_INT32I:
    strm << "32bit INT";
    break;

  case BUFFER_DATA_TYPE_INT64I:
    strm << "64 bit INT";
    break;

  case BUFFER_DATA_TYPE_INT8:
    strm << "8 bit INT";
    break;

  case BUFFER_DATA_TYPE_INVAL:
    strm << "INVALID";
    break;
  }
  strm << "\n";

  return strm.str();
}

[[nodiscard]] std::string
to_string(const qaic::rt::BufferMappings &allBufferMappings) {
  std::ostringstream strm;
  for (const auto &bufMap : allBufferMappings) {
    std::cout << to_string(bufMap);
  }

  return strm.str();
}

[[nodiscard]] std::string to_string(const std::vector<int64_t> &tokenVec) {
  if (tokenVec.empty()) {
    return "";
  }

  std::ostringstream strm;
  std::copy(tokenVec.begin(), tokenVec.end() - 1,
            std::ostream_iterator<int64_t>(strm, ", "));
  strm << tokenVec.back();

  strm << "\n";
  return strm.str();
}

class QBufferWrapper {
public:
  explicit QBufferWrapper(size_t size) : buffer_{size, new uint8_t[size]} {}
  ~QBufferWrapper() { delete[] buffer_.buf; }

  [[nodiscard]] QBuffer &getQBuffer() { return buffer_; }

private:
  QBuffer buffer_;
};
using shQBufferWrapper = std::shared_ptr<QBufferWrapper>;

[[nodiscard]] shQBufferWrapper
createBuffer(const std::string &bufferName,
             const qaic::rt::BufferMappings &allBufferMappings) {
  auto it =
      std::find_if(allBufferMappings.begin(), allBufferMappings.end(),
                   [&bufferName](const qaic::rt::BufferMapping &bufferMapping) {
                     return (bufferName == bufferMapping.bufferName);
                   });

  if (it != allBufferMappings.end()) {
    return std::make_shared<QBufferWrapper>(it->size);
  }

  throw std::runtime_error(
      "Buffer mapping of Input Type not found for buffer named : " +
      bufferName);
}

template <typename T>
[[nodiscard]] std::string qBufferToString(shQBufferWrapper wrappedBuf) {
  std::ostringstream strm;
  auto rawBufPtr = wrappedBuf->getQBuffer().buf;
  const T *bufferT = reinterpret_cast<const T *>(rawBufPtr);
  int numT = wrappedBuf->getQBuffer().size / sizeof(T);
  for (int i = 0; i < numT; i++) {
    strm << "[ " << i << " ] = " << bufferT[i] << "\n";
  }
  return strm.str();
}

void populateInputIdBuffer(QBuffer &inputIdBuffer,
                           const std::vector<int64_t> &tokenVector) {
  std::copy_n(reinterpret_cast<const uint8_t *>(tokenVector.data()),
              tokenVector.size() * sizeof(int64_t), inputIdBuffer.buf);
  std::fill(inputIdBuffer.buf + tokenVector.size() * sizeof(int64_t),
            inputIdBuffer.buf + inputIdBuffer.size -
                tokenVector.size() * sizeof(int64_t),
            0);
}

void populateAttentionMaskBuffer(QBuffer &attentionMaskBuffer,
                                 const std::vector<int64_t> &tokenVector) {
  std::fill(attentionMaskBuffer.buf,
            attentionMaskBuffer.buf + attentionMaskBuffer.size, 0);
  for (unsigned int i = 0; i < tokenVector.size(); ++i) {
    int64_t *bufferChunkInt = reinterpret_cast<int64_t *>(
        attentionMaskBuffer.buf + (i * sizeof(int64_t)));
    *bufferChunkInt = 1;
  }
}

[[nodiscard]] std::string
getPredictedWordFromOutputBuffer(shQBufferWrapper &outputBuffer,
                                 int64_t maskIndex,
                                 shVocabularyHelper vocabHelper) {
  const float *bufferFloat =
      reinterpret_cast<const float *>(outputBuffer->getQBuffer().buf);
  int64_t numLogitsPerToken = vocabHelper->getVocabularyTable().size();
  auto startIter = bufferFloat + (numLogitsPerToken * maskIndex);
  auto endIter = startIter + numLogitsPerToken;
  auto maxElemIter = std::max_element(startIter, endIter);
  auto maxIndex = static_cast<int64_t>(std::distance(startIter, maxElemIter));
  auto word = vocabHelper->getWordAtIndex(maxIndex);
  return word;
}

} // namespace cloudai

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
  using namespace cloudai;
  constexpr const char *qpcPath =
      "/work_dir/examples_work/cloud-ai/tutorials/NLP/"
      "Model-Onboarding-Beginner/bert-base-cased/generatedModels/"
      "bert-base-cased_fix_outofrange_fp16_qpc";
  auto qpc = qaic::rt::Qpc::Factory(qpcPath);

  const auto &allBufferMappings = qpc->getBufferMappings();
  // std::cout << "Buffer Mappings : \n" << to_string(allBufferMappings);

  constexpr const char *vocabFilePath =
      "/work_dir/examples_work/cpp_examples_work/hugging_face_data/vocab.txt";
  shVocabularyHelper vocabHelper =
      std::make_shared<VocabularyHelper>(vocabFilePath);
  // std::cout << "Size of vocabulary = " <<
  // vocabHelper->getVocabularyTable().size() << std::endl;

  Tokenizer tokenizer(vocabHelper);
  const std::string inputSentence =
      "[CLS] [MASK] is the capital of France [SEP]";
  // const std::string inputSentence = "[CLS] The dog [MASK] on the mat .
  // [SEP]";
  int64_t maskIndex = 1;
  auto const inputTokens = tokenizer.getTokensForSentence(inputSentence);
  // std::cout << "Input Tokens : " << to_string(inputTokens);

  auto inputIdBuffer = createBuffer("input_ids", allBufferMappings);
  populateInputIdBuffer(inputIdBuffer->getQBuffer(), inputTokens);
  // std::cout << "Input Id Buffer : \n" <<
  // qBufferToString<int64_t>(inputIdBuffer);

  auto attentionMaskBuffer = createBuffer("attention_mask", allBufferMappings);
  populateAttentionMaskBuffer(attentionMaskBuffer->getQBuffer(), inputTokens);
  // std::cout << "Attendion Mask Buffer : \n" <<
  // qBufferToString<int64_t>(attentionMaskBuffer);

  auto outputBuffer = createBuffer("logits", allBufferMappings);
  std::fill(outputBuffer->getQBuffer().buf,
            outputBuffer->getQBuffer().buf + outputBuffer->getQBuffer().size,
            0);
  // std::cout << "Output Buffer (initialized): \n" <<
  // qBufferToString<float>(outputBuffer);

  std::vector<QID> qidList{0};
  auto context = qaic::rt::Context::Factory(nullptr, qidList);

  // *** INFERENCE SET ***
  constexpr uint32_t setSize = 1;
  constexpr uint32_t numActivations = 1;
  auto inferenceSet = qaic::rt::InferenceSet::Factory(
      context, qpc, qidList.at(0), setSize, numActivations);

  // *** SETUP IO BUFFERS ***
  qaic::rt::shInferenceHandle submitHandle;
  auto status = inferenceSet->getAvailable(submitHandle);
  if (status != QS_SUCCESS) {
    std::cerr << "Error obtaining Inference Handle\n";
    return -1;
  }

  std::vector<QBuffer> inputBuffers{inputIdBuffer->getQBuffer(),
                                    attentionMaskBuffer->getQBuffer()};
  std::vector<QBuffer> outputBuffers{outputBuffer->getQBuffer()};
  submitHandle->setInputBuffers(inputBuffers);
  submitHandle->setOutputBuffers(outputBuffers);

  // *** SUBMIT ***
  constexpr uint32_t inferenceId = 0; // also named as request ID
  status = inferenceSet->submit(submitHandle, inferenceId);
  if (status != QS_SUCCESS) {
    std::cerr << "Error in submitting handle through InferenceSet\n";
    return -1;
  }

  // *** COMPLETION ***
  qaic::rt::shInferenceHandle completedHandle;
  status = inferenceSet->getCompletedId(completedHandle, inferenceId);
  if (status != QS_SUCCESS) {
    std::cerr << "Error in getting completed handle through InferenceSet\n";
    return -1;
  }

  status = inferenceSet->putCompleted(std::move(completedHandle));
  if (status != QS_SUCCESS) {
    std::cerr << "Error in putting completed handle through InferenceSet\n";
    return -1;
  }

  // std::cout << "Output Buffer (populated): \n" <<
  // qBufferToString<float>(outputBuffer);
  auto predictedWord =
      getPredictedWordFromOutputBuffer(outputBuffer, maskIndex, vocabHelper);
  std::cout << "Input Sentence : " << inputSentence << std::endl;
  std::cout << "Predicted Word = " << predictedWord << std::endl;
}