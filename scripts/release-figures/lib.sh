# Checks out <sha> as a detached worktree at <dir>, fetching it from the
# gate repository the job runs in, or from the public repository when the
# gate repository does not have it.
checkout_sha() {
  { git fetch -q --depth=1 origin "$1" || git fetch -q --depth=1 https://github.com/Anyesh/incr.git "$1"; } &&
    git worktree add -q --detach "$2" FETCH_HEAD
}
