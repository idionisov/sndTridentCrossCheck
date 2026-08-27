#pragma once

#include <vector>

#include "sndBaseCut.h"

#include "TChain.h"
#include "TClonesArray.h"
#include "sndScifiHit.h"

namespace snd {
  namespace trident_cuts {
  
    class sciFiBaseCut : public snd::trident_cuts::baseCut {

    private : 
      static TChain * tree;
      static unsigned long int read_entry;

    protected :
      static TClonesArray * scifiDigiHitCollection;

      static std::vector<int> hits_per_plane_vertical;
      static std::vector<int> hits_per_plane_horizontal;
      static std::vector<double> signal_per_plane_vertical;
      static std::vector<double> signal_per_plane_horizontal;

      void initializeEvent();

    public :
      sciFiBaseCut(TChain * ch);
      virtual ~sciFiBaseCut(){;}
      static void setupBranch(TChain * ch);
      static TClonesArray * getSciFiDigiHitCollection() { return scifiDigiHitCollection; }
    };

  }
}
