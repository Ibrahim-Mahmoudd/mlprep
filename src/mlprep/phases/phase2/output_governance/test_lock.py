"""Test-partition locking for Phase 2."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from mlprep.phases.phase2.output_governance.metadata import (
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
    LockedTestPartition,
    SplitFinding,
)
from mlprep.phases.phase2.output_governance.state import (
    Phase2State,
)


@dataclass(slots=True)
class TestLockOptions:
    """Options controlling test-partition locking."""

    enabled: bool = True
    require_access_reason: bool = True

    generate_lock_token: bool = True
    lock_token_length: int = 32

    def __post_init__(self) -> None:
        """Validate test-lock options."""

        if self.lock_token_length < 16:
            raise ValueError(
                "lock_token_length must be at least 16."
            )


class TestPartitionLocker:
    """Lock and govern access to the test partition."""

    component_name = "test_lock"

    def __init__(
        self,
        options: TestLockOptions | None = None,
    ) -> None:
        """Initialize the test locker."""

        self.options = options or TestLockOptions()

    def lock(
        self,
        *,
        test: DatasetPartition,
        state: Phase2State,
    ) -> tuple[
        DatasetPartition | LockedTestPartition,
        str | None,
        list[SplitFinding],
    ]:
        """Lock the test partition."""

        if not self.options.enabled:
            state.test_lock.locked = False

            return test, None, []

        token: str | None = None
        token_hash: str | None = None

        if self.options.generate_lock_token:
            token = secrets.token_urlsafe(
                self.options.lock_token_length
            )

            token_hash = hashlib.sha256(
                token.encode("utf-8")
            ).hexdigest()

        state.test_lock.mark_locked(
            lock_token_hash=token_hash
        )

        test.metadata.is_locked = True

        finding = SplitFinding(
            code="P2-TEST-LOCK-001",
            category=SplitFindingCategory.TEST_LOCK,
            severity=SplitFindingSeverity.INFO,
            component=self.component_name,
            message=(
                "The test partition was locked successfully."
            ),
            recommended_action=(
                SplitFindingAction.LOCK_TEST_SET
            ),
            applied_action=(
                SplitFindingAction.LOCK_TEST_SET
            ),
            blocks_readiness=False,
            details={
                "token_generated": (
                    token is not None
                ),
            },
        )

        return (
            LockedTestPartition(test),
            token,
            [finding],
        )

    def authorize_access(
        self,
        *,
        locked_test: LockedTestPartition,
        state: Phase2State,
        reason: str,
        token: str | None = None,
    ) -> DatasetPartition:
        """Authorize access to a locked test partition."""

        if (
            self.options.require_access_reason
            and not reason.strip()
        ):
            raise ValueError(
                "A test-access reason is required."
            )

        expected_hash = (
            state.test_lock.lock_token_hash
        )

        if expected_hash is not None:
            if token is None:
                raise PermissionError(
                    "A test lock token is required."
                )

            provided_hash = hashlib.sha256(
                token.encode("utf-8")
            ).hexdigest()

            if not secrets.compare_digest(
                expected_hash,
                provided_hash,
            ):
                raise PermissionError(
                    "The test lock token is invalid."
                )

        state.test_lock.mark_access(reason)

        return locked_test.access(
            reason=reason,
            allow=True,
        )
