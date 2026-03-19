from huggingface_hub import HfApi

def create_repo():
    api = HfApi()
    api.create_repo( "KalooIna/cybersecurity_attacks" , repo_type = "model" )


def upload_model_to_huggingface():    
    api = HfApi()
    if not api.repo_exists( "KalooIna/cybersecurity_attacks" , repo_type = "model" ) :
        create_repo()
    # Carica i modelli
    api.upload_file(
        path_or_fileobj = "models/extra_trees_model.pkl" ,
        path_in_repo = "models/extra_trees_model.pkl" ,
        repo_id = "KalooIna/cybersecurity_attacks" ,
    )
    api.upload_file(
        path_or_fileobj = "models/logit_model.pkl" ,
        path_in_repo = "models/logit_model.pkl" ,
        repo_id = "KalooIna/cybersecurity_attacks" ,
    )
    api.upload_file(
        path_or_fileobj = "models/randomforrest_model.pkl" ,
        path_in_repo = "models/randomforrest_model.pkl" ,
        repo_id = "KalooIna/cybersecurity_attacks" ,
    )
