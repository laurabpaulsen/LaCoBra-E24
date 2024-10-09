import mne
import numpy as np
import pandas as pd
from pathlib import Path


# function for preprocessing one subject
def preproc_subject(raw, logfile, event_id, bad_channels = ["Fp1", "Fp2"], reject = {"eeg":150e-6}, l_freq = 1, h_freq = 40, tmin = -0.2, tmax = 1, baseline = (-0.2, None)):
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
    events, _ = mne.events_from_annotations(raw, verbose=False)
    events = get_triggers(logfile, events)

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

            # SHIT SHOW
            if event[-1] == 20:
                congruency = "congruent" if target_gender == prime_gender else "incongruent" 
                
                if prime_gender in ["female", "male"] and target_gender in ["female", "male"]:
                    tmp_event[-1] = event_id[f"word/target/{prime_gender}/{congruency}"]
                
                elif prime_gender == "filler":
                    tmp_event[-1] = event_id[f"word/target/{prime_gender}"]
                
                else:
                    tmp_event[-1] = event_id[f"word/target/{prime_gender}/{target_gender}"]

        
            # WORKS
            if event[-1] == 30:
                congruency = "congruent" if target_gender == prime_gender else "incongruent" 

                if prime_gender == "filler":
                    tmp_event[-1] = event_id[f"response/{correct}/{prime_gender}/{response}"]
                elif prime_gender in ["female", "male"] and target_gender in ["female", "male"]:
                    tmp_event[-1] = event_id[f"response/{correct}/{prime_gender}/{congruency}/{response}"]
                else:
                    tmp_event[-1] = event_id[f"response/{correct}/{prime_gender}/{target_gender}/{response}"]
        
            updated_events.append(tmp_event)


    updated_events = np.array(updated_events)
    
    return updated_events
