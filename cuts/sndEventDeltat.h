#pragma once

#include "sndEventHeaderBaseCut.h"

#include "TChain.h"

namespace snd {
  namespace analysis_cuts {

    class eventDeltatCut : public snd::analysis_cuts::EventHeaderBaseCut {
    private:
      int delta_t;
      int delta_e;
    public :
      eventDeltatCut(int delta_event, int delta_timestamp, TChain * ch);
      ~eventDeltatCut(){;}
      bool passCut();
    };

  }
}
