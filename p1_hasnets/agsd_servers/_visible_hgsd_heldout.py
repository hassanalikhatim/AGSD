# from .server_hasnet_from_heldout import Server_HaSNet_from_HeldOut

# from .server_hasnet_from_heldout_good_two import Server_HaSNet_from_HeldOut


"""
This server had a very dumb logical error, so it was giving very good results. Please don't trust this server.
"""
# from .server_hasnet_error import Server_HaSNet_from_HeldOut


"""
This HGSD server cluster client submissions into several clusters and heals each individual cluster. 
The server then compares the change in the standard deviation of cluster aggregated models before and
after healing.
"""
# from .hgsd_datacentric_healing import Server_HaSNet_from_HeldOut


"""
This HGSD server adversarially attacks individual client submissions and compares the standard deviation 
and classification confidence of the submissions. Backdoored clients show small standard deviation (becuase 
most adversarial samples are classified to the backdoor attack target class) and high confidence (because 
backdoor attacks are very effective).
"""
# from .hgsd_heldout_datacentric import Server_HaSNet_from_HeldOut


"""
This HGSD server is similar to the one in .hgsd_heldout_datacentric, but comparatively much efficient. 
This is because it carries out transfer attacks from aggregated model to the client models insteald of 
attacking client models individually.
"""
from .agsd_id import AGSD_ID


"""
This HGSD server is similar to the one in .hgsd_heldout_datacentric_transfer, but comparatively more efficient. 
This is because this server computes the aggregated models of each cluster and computed the standard deviation and
confidences over the aggregated models of clusters instead of each client model individually. This also gives you a
hyperparameter (the number of preliminary clusters) that we can tune according to our requirements for an 
efficiency-performance tradeoff. 
"""
# from .agsd_id_efficient import Server_HaSNet_from_HeldOut


