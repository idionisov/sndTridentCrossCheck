#pragma once

#include "sndSciFiBaseCut.h"

namespace snd::analysis_cuts {

  class minMaxSciFiSignalCut : public sciFiBaseCut {
  private:
    double signal_threshold;

  public:
    minMaxSciFiSignalCut(double threshold, TChain * ch);
    ~minMaxSciFiSignalCut() { ; }

    bool passCut() override;
  };

}
