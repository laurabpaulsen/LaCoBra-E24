import mne
import numpy as np
import pandas as pd
from pathlib import Path


# function for preprocessing one subject
def preproc_subject(raw, event_id, events, bad_channels = ["Fp1", "Fp2"], reject = {"eeg":150e-6}, l_freq = 1, h_freq = 40, tmin = -0.2, tmax = 1, baseline = (-0.2, None)):
    """
    Preprocesses one subject's data by setting the montage, removing and interpolating bad channels, filtering and creating epochs

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    """
    # set montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    
    # remove bad channels
    raw.info["bads"] = bad_channels
    raw.interpolate_bads(reset_bads = True)
    
    # bandpass filter
    raw.filter(l_freq, h_freq)
    
    # epoching
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline = baseline, reject = reject, preload = True)
    
    # artifact rejection
    epochs.drop_bad()
    
    return epochs

# function for getting only the event_ids found in the given session
def update_event_ids(events, event_id):
    """
    This function updates the event_id dictionary to only contain mappings between things that actually happend in the experiment for your given participant
    """
    # modify event id so we only have triggers from our data so we do not get an error when we are trying to run 
    triggers_unique = np.unique(events[:, 2])

    # Create a new dictionary for the selected triggers
    updated_event_ids = {}

    # Iterate through the event_id dictionary
    for key, value in event_id.items():
        if value in triggers_unique:
            updated_event_ids[key] = value

    return updated_event_ids


def update_events_group1_group2(events, event_id, logfile):

    updated_events = []

    counter = -1

    for event in events:
        tmp_event = event

        if event[-1] in [10, 20, 30]:
            if event[-1] == 10:
                counter += 1
            
            prime_gender = logfile.iloc[counter]["prime_gender"]
            prime_age = logfile.iloc[counter]["prime_age"]
            target_gender = logfile.iloc[counter]["target_gender"]
            #print(prime_gender, target_gender)
            word_type = logfile.iloc[counter]["type"]
            correct = logfile.iloc[counter]["correct"]
            correct = "correct" if correct == 1 else "incorrect"
            response = logfile.iloc[counter]["response"] # m or z


            ## WORKS
            if event[-1] == 10:
                if prime_gender == "filler":
                    tmp_event[-1] = event_id[f"word/prime/{prime_gender}"]
                else:
                    tmp_event[-1] = event_id[f"word/prime/{prime_gender}/{prime_age}"]

            if event[-1] == 20:
                congruency = "congruent" if target_gender == prime_gender else "incongruent" 
                
                if prime_gender in ["female", "male"] and target_gender in ["female", "male"]:
                    tmp_event[-1] = event_id[f"word/target/{target_gender}/{congruency}"]
                
                else:
                    tmp_event[-1] = event_id[f"word/target/{target_gender}/{prime_gender}"]

            if event[-1] == 30:
                congruency = "congruent" if target_gender == prime_gender else "incongruent" 

                if target_gender in ["control", "neutral"]:
                    tmp_event[-1] = event_id[f"response/{correct}/{target_gender}/{prime_gender}/{response}"]
                
                elif prime_gender in ["female", "male"] and target_gender in ["female", "male"]:
                    tmp_event[-1] = event_id[f"response/{correct}/{target_gender}/{congruency}/{response}"]
                
                else:
                    tmp_event[-1] = event_id[f"response/{correct}/{target_gender}/{prime_gender}/{response}"]
        
            updated_events.append(tmp_event)


    updated_events = np.array(updated_events)
    
    return updated_events


event_id = {
   # word / prime / prime_gender / age
   'word/prime/female/adult': 11,
   'word/prime/female/child': 12,
   'word/prime/female/neutral': 13,
   'word/prime/male/adult': 21,
   'word/prime/male/child': 22,
   'word/prime/male/neutral': 23,
   'word/prime/neutral/adult': 31,
   'word/prime/neutral/child': 32,
   'word/prime/neutral/neutral': 33,
   'word/prime/filler': 40,

   # word / target / target_gender / congruency or filler or neutral (prime gender)
   'word/target/female/congruent': 111,
   'word/target/female/incongruent': 121,
   'word/target/female/neutral': 131,
   'word/target/female/filler': 141,
   'word/target/male/incongruent': 112,
   'word/target/male/congruent': 122,
   'word/target/male/neutral': 132,
   'word/target/male/filler': 142,
   'word/target/neutral/female': 113,
   'word/target/neutral/male': 123,
   'word/target/neutral/neutral': 133,
   'word/target/neutral/filler': 143,
   'word/target/control/female': 114,
   'word/target/control/male': 124,
   'word/target/control/neutral': 134,
   'word/target/control/filler': 144,

   # response / correct / target_gender / congruency or filler or neutral (prime gender) / button that was pressed
   'response/incorrect/female/congruent/m': 161,
   'response/incorrect/female/congruent/z': 166,
   'response/incorrect/female/incongruent/m': 171,
   'response/incorrect/female/incongruent/z': 176,
   'response/incorrect/female/neutral/m': 181,
   'response/incorrect/female/neutral/z': 186,
   'response/incorrect/female/filler/m': 191,
   'response/incorrect/female/filler/z': 196,
   'response/incorrect/male/incongruent/m': 162,
   'response/incorrect/male/incongruent/z': 167,
   'response/incorrect/male/congruent/m': 172,
   'response/incorrect/male/congruent/z': 177,
   'response/incorrect/male/neutral/m': 182,
   'response/incorrect/male/neutral/z': 187,
   'response/incorrect/male/filler/m': 192,
   'response/incorrect/male/filler/z': 197,
   'response/incorrect/neutral/female/m': 163,
   'response/incorrect/neutral/female/z': 168,
   'response/incorrect/neutral/male/m': 173,
   'response/incorrect/neutral/male/z': 178,
   'response/incorrect/neutral/neutral/m': 183,
   'response/incorrect/neutral/neutral/z': 188,
   'response/incorrect/neutral/filler/m': 193,
   'response/incorrect/neutral/filler/z': 198,
   'response/incorrect/control/female/m': 164,
   'response/incorrect/control/female/z': 169,
   'response/incorrect/control/male/m': 174,
   'response/incorrect/control/male/z': 179,
   'response/incorrect/control/neutral/m': 184,
   'response/incorrect/control/neutral/z': 189,
   'response/incorrect/control/filler/m': 194,
   'response/incorrect/control/filler/z': 199,
   'response/correct/female/congruent/m': 211,
   'response/correct/female/congruent/z': 216,
   'response/correct/female/incongruent/m': 221,
   'response/correct/female/incongruent/z': 226,
   'response/correct/female/neutral/m': 231,
   'response/correct/female/neutral/z': 236,
   'response/correct/female/filler/m': 241,
   'response/correct/female/filler/z': 246,
   'response/correct/male/incongruent/m': 212,
   'response/correct/male/incongruent/z': 217,
   'response/correct/male/congruent/m': 222,
   'response/correct/male/congruent/z': 227,
   'response/correct/male/neutral/m': 232,
   'response/correct/male/neutral/z': 237,
   'response/correct/male/filler/m': 242,
   'response/correct/male/filler/z': 247,
   'response/correct/neutral/female/m': 213,
   'response/correct/neutral/female/z': 218,
   'response/correct/neutral/male/m': 223,
   'response/correct/neutral/male/z': 228,
   'response/correct/neutral/neutral/m': 233,
   'response/correct/neutral/neutral/z': 238,
   'response/correct/neutral/filler/m': 243,
   'response/correct/neutral/filler/z': 248,
   'response/correct/control/female/m': 214,
   'response/correct/control/female/z': 219,
   'response/correct/control/male/m': 224,
   'response/correct/control/male/z': 229,
   'response/correct/control/neutral/m': 234,
   'response/correct/control/neutral/z': 239,
   'response/correct/control/filler/m': 244,
   'response/correct/control/filler/z': 249
}