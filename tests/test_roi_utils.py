import numpy as np
import pytest
import aind_ophys_utils.roi_utils as ru


@pytest.mark.parametrize(
    "n_masks",
    [
        1,
        2,
        5,
        10,
        100,
    ],
)
def test_mask_list_from_array(n_masks):
    """Test mask_list_from_array"""

    masks_array = np.zeros((512, 512))

    # add squares of size 20 at random locations, with values 1-5
    for i in range(n_masks):
        x = np.random.randint(0, 512 - 20)
        y = np.random.randint(0, 512 - 20)
        masks_array[x : x + 20, y : y + 20] = i + 1

    mask_list, _ = ru.mask_list_from_array(masks_array)

    assert len(mask_list) == n_masks
    # each mask in list only has one unique value (0 and N)
    for mask in mask_list:
        assert len(np.unique(mask)) == 2
