import os
import shutil

def get_log_name(model, pred_len, epochs, job_id="", suffix=""):
    """
    Generate a descriptive log filename based on model type and architecture parameters.
    
    Format: {model_type}_d{d_model}_h{nhead}_l{num_encoder_layers}_p{pred_len}_e{epochs}{suffix}.log
    
    Example outputs:
        - tptrans_d512_h8_l4_p24_e200_final.log
        - informer_d256_h4_l3_p12_e100_job12345.log
    
    Args:
        model: Model instance
        pred_len (int): Prediction length
        epochs (int): Number of training epochs
        job_id (str): Optional job ID to include (e.g., LSF job ID)
        suffix (str): Optional suffix (e.g., "_final", "_train")
    
    Returns:
        str: Log filename without the .log extension
    """
    model_class = model.__class__.__name__
    model_type = model_class.lower()
    
    d_model = getattr(model, 'd_model', None)
    nhead = getattr(model, 'nhead', None)
    num_encoder_layers = getattr(model, 'num_encoder_layers', None)
    
    if d_model is None or nhead is None or num_encoder_layers is None:
        raise ValueError(
            f"Model {model_class} must have the following attributes stored: "
            f"d_model, nhead, num_encoder_layers. "
            f"Got d_model={d_model}, nhead={nhead}, num_encoder_layers={num_encoder_layers}"
        )
    
    if job_id:
        job_part = f"_job{job_id}"
    else:
        job_part = ""
    
    filename = f"{model_type}_d{d_model}_h{nhead}_l{num_encoder_layers}_p{pred_len}_e{epochs}{job_part}{suffix}"
    return filename


def rename_output_log(model, pred_len, epochs, original_log_path, log_dir="logs"):
    """
    Rename and move an output log file to include model parameters.
    
    Creates a 'logs' directory if it doesn't exist and moves the log file there with
    a descriptive name based on model architecture.
    
    Args:
        model: Model instance
        pred_len (int): Prediction length
        epochs (int): Number of training epochs
        original_log_path (str): Path to the original log file (e.g., "output_26957291.log")
        log_dir (str): Directory to move log to (default: "logs")
    
    Returns:
        str: Full path to the renamed and moved log file
    
    Example:
        >>> new_path = rename_output_log(model, pred_len=24, epochs=100, 
        ...                               original_log_path="output_26957291.log")
        >>> print(f"Log moved to: {new_path}")
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Extract job ID from original log path if available
    job_id = ""
    if "output_" in original_log_path:
        parts = original_log_path.split("output_")
        if len(parts) > 1:
            job_id = parts[1].replace(".log", "").strip()
    
    log_name = get_log_name(model, pred_len, epochs, job_id=job_id, suffix="")
    new_log_path = os.path.join(log_dir, f"{log_name}.log")
    
    if os.path.exists(original_log_path):
        shutil.move(original_log_path, new_log_path)
    
    return new_log_path

