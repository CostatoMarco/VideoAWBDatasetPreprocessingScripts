/*
 * Copyright 2023 MotionCam
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>

#include <motioncam/Decoder.hpp>
#include <audiofile/AudioFile.h>
#include <filesystem>
namespace fs = std::filesystem;


#define TINY_DNG_WRITER_IMPLEMENTATION
    #include <tinydng/tiny_dng_writer.h>
#undef TINY_DNG_WRITER_IMPLEMENTATION

void writeAudio(
    const std::string& outputPath,
    const int sampleRateHz,
    const int numChannels,
    std::vector<motioncam::AudioChunk>& audioChunks)
{
    AudioFile<int16_t> audio;
    
    audio.setNumChannels(numChannels);
    audio.setSampleRate(sampleRateHz);
    
    if(numChannels == 2) {
        for(auto& x : audioChunks) {
            for(auto i = 0; i < x.second.size(); i+=2) {
                audio.samples[0].push_back(x.second[i]);
                audio.samples[1].push_back(x.second[i+1]);
            }
        }
    }
    else if(numChannels == 1) {
        for(auto& x : audioChunks) {
            for(auto i = 0; i < x.second.size(); i++)
                audio.samples[0].push_back(x.second[i]);
        }
    }
    
    audio.save(outputPath);
}

void writeDng(
    const std::string& outputPath,
    const std::vector<uint16_t>& data,
    const nlohmann::json& metadata,
    const nlohmann::json& containerMetadata)
{
    const unsigned int width = metadata["width"];
    const unsigned int height = metadata["height"];
    
    std::vector<double> asShotNeutral = metadata["asShotNeutral"];

    std::vector<uint16_t> blackLevel = containerMetadata["blackLevel"];
    double whiteLevel = containerMetadata["whiteLevel"];
    std::string sensorArrangement = containerMetadata["sensorArrangment"];
    std::vector<double> colorMatrix1 = containerMetadata["colorMatrix1"];
    std::vector<double> colorMatrix2 = containerMetadata["colorMatrix2"];
    std::vector<double> forwardMatrix1 = containerMetadata["forwardMatrix1"];
    std::vector<double> forwardMatrix2 = containerMetadata["forwardMatrix2"];

    // Create first frame
    tinydngwriter::DNGImage dng;

    dng.SetBigEndian(false);
    dng.SetDNGVersion(0, 0, 4, 1);
    dng.SetDNGBackwardVersion(0, 0, 1, 1);
    dng.SetImageData(reinterpret_cast<const unsigned char*>(data.data()), data.size());
    dng.SetImageWidth(width);
    dng.SetImageLength(height);
    dng.SetPlanarConfig(tinydngwriter::PLANARCONFIG_CONTIG);
    dng.SetPhotometric(tinydngwriter::PHOTOMETRIC_CFA);
    dng.SetRowsPerStrip(height);
    dng.SetSamplesPerPixel(1);
    dng.SetCFARepeatPatternDim(2, 2);
    
    dng.SetBlackLevelRepeatDim(2, 2);
    dng.SetBlackLevel(4, blackLevel.data());
    dng.SetWhiteLevelRational(1, &whiteLevel);

    std::vector<uint8_t> cfa;
    
    if(sensorArrangement == "rggb")
        cfa = { 0, 1, 1, 2 };
    else if(sensorArrangement == "bggr")
        cfa = { 2, 1, 1, 0 };
    else if(sensorArrangement == "grbg")
        cfa = { 1, 0, 2, 1 };
    else if(sensorArrangement == "gbrg")
        cfa = { 1, 2, 0, 1 };
    else
        throw std::runtime_error("Invalid sensor arrangement");

    dng.SetCFAPattern(4, cfa.data());
    
    // Rectangular
    dng.SetCFALayout(1);

    const uint16_t bps[1] = { 16 };
    dng.SetBitsPerSample(1, bps);
    
    dng.SetColorMatrix1(3, colorMatrix1.data());
    dng.SetColorMatrix2(3, colorMatrix2.data());

    dng.SetForwardMatrix1(3, forwardMatrix1.data());
    dng.SetForwardMatrix2(3, forwardMatrix2.data());
    
    dng.SetAsShotNeutral(3, asShotNeutral.data());
    
    const uint32_t activeArea[4] = { 0, 0, height, width };
    dng.SetActiveArea(&activeArea[0]);

    // Write DNG
    std::string err;
    tinydngwriter::DNGWriter writer(false);

    writer.AddImage(&dng);

    writer.WriteToFile(outputPath.c_str(), &err);
}

int main(int argc, const char * argv[]) {
    if(argc < 2) {
        std::cout << "Usage: decoder <input file> [-n number of frames to export]" << std::endl;
        return -1;
    }
    
   fs::path inputDir{argv[1]};
    if (!fs::is_directory(inputDir)) {
        std::cerr << "Error: '" << inputDir << "' is not a directory.\n";
        return 1;
    }
    for (auto& entry : fs::directory_iterator(inputDir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".mcraw") continue;

        // Create output subdirectory
        auto baseName = entry.path().stem();
        fs::path outDir = inputDir / baseName;
        fs::create_directories(outDir);

        try {
            // Decode
            motioncam::Decoder decoder(entry.path().string());
            auto frames = decoder.getFrames();
            auto containerMetadata = decoder.getContainerMetadata();

            // --- Audio ---
            std::vector<motioncam::AudioChunk> audioChunks;
            decoder.loadAudio(audioChunks);
            writeAudio(outDir / "audio.wav",
                       decoder.audioSampleRateHz(),
                       decoder.numAudioChannels(),
                       audioChunks);

            // --- Video frames ---
            std::vector<uint16_t> frameData;
            nlohmann::json frameMeta;
            for (size_t i = 0; i < frames.size(); ++i) {
                decoder.loadFrame(frames[i], frameData, frameMeta);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "frame_%06zu.dng", i);
                writeDng(outDir / buf, frameData, frameMeta, containerMetadata);
            }

            std::cout << "Decoded '" << entry.path().filename()
                      << "' into '" << outDir << "/'.\n";
        }
        catch (const motioncam::MotionCamException& e) {
            std::cerr << "Failed to decode '"
                      << entry.path().filename()
                      << "': " << e.what() << "\n";
        }
    }

    
    
    
    return 0;
}
