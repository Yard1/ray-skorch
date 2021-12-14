from skorch.callbacks.logging import filter_log_keys


class SortedKeysMixin:
    def _sorted_keys(self, keys, keys_ignored=None, filter_keys=True):
        """Sort keys, dropping the ones that should be ignored.

        The keys that are in ``self.ignored_keys`` or that end on
        '_best' are dropped. Among the remaining keys:
          * 'epoch' is put first;
          * 'dur_s' is put last;
          * keys that start with 'event_' are put just before 'dur_s';
          * all remaining keys are sorted alphabetically.

        This is a copy of a skorch method, modified to replace
        'dur' with 'dur_s' and to allow for toggling whether
        '*_best' or 'event_*' keys should be filtered.
        """
        sorted_keys = []
        keys_ignored = keys_ignored or {}

        # make sure "epoch" comes first
        if ("epoch" in keys) and ("epoch" not in keys_ignored):
            sorted_keys.append("epoch")

        # ignore keys like *_best or event_*
        if filter_keys:
            for key in filter_log_keys(
                    sorted(keys), keys_ignored=keys_ignored):
                if key != "dur_s":
                    sorted_keys.append(key)
        else:
            sorted_keys.extend(
                sorted([
                    key for key in keys
                    if key not in keys_ignored and not key.startswith("event_")
                ]))

        # add event_* keys
        for key in sorted(keys):
            if key.startswith("event_") and (key not in keys_ignored):
                sorted_keys.append(key)

        # make sure "dur" comes last
        if ("dur_s" in keys) and ("dur_s" not in keys_ignored):
            sorted_keys.append("dur_s")

        return sorted_keys
