"""Roles, permissions, PHI masking and separation of duties (build brief §13).

    "Eight roles from §9... PHI masking by default with an explicit,
     audited unmask action... Override as a DISTINCT permission, not an
     implicit consequence of seniority... ADMIN-only kill switch and rollback."

The design point the tests keep returning to is §13.2's distinction: mask
IDENTITY, not EVIDENCE. A reviewer who cannot name a provider must still be
able to tell two providers apart, follow one across pages and describe "the
provider in case X" — so the mask is a *stable pseudonym*, never a blackout.
"""

from __future__ import annotations

import pytest

from fwa.auth.rbac import (
    AccessDenied, PAGE_PERMISSIONS, Permission, ROLE_PERMISSIONS, Role,
    SeparationOfDutiesError, assert_separation_of_duties, has_permission, mask_frame,
    mask_identity, require, visible_pages,
)

ALL_ROLES = list(Role)


# ------------------------------------------------------------------ the roles


def test_there_are_exactly_the_eight_roles_from_section_nine():
    assert {r.value for r in Role} == {
        "ADMIN", "POLICY_OWNER", "CLAIMS_REVIEWER", "CLINICAL_REVIEWER",
        "SIU_INVESTIGATOR", "ANALYST", "QA", "AUDITOR",
    }


def test_every_role_has_a_permission_set():
    for role in ALL_ROLES:
        assert role in ROLE_PERMISSIONS
        assert ROLE_PERMISSIONS[role], f"{role.value} holds no permissions at all."


def test_no_role_except_admin_holds_every_permission():
    for role in ALL_ROLES:
        if role is Role.ADMIN:
            continue
        assert ROLE_PERMISSIONS[role] != set(Permission)


def test_every_page_is_gated_by_a_permission():
    assert len(PAGE_PERMISSIONS) >= 10
    for page, permission in PAGE_PERMISSIONS.items():
        assert isinstance(permission, Permission), page


def test_each_role_sees_a_coherent_subset_of_pages():
    for role in ALL_ROLES:
        pages = visible_pages(role)
        assert pages, f"{role.value} can see no pages at all."
        assert set(pages) <= set(PAGE_PERMISSIONS)


def test_an_auditor_can_read_but_not_act():
    """The read-only role is the clearest test that permissions are not cosmetic."""
    assert has_permission(Role.AUDITOR, Permission.VIEW_AUDIT_LOG)
    for write in (Permission.RECORD_DISPOSITION, Permission.AUTHOR_RULE,
                  Permission.KILL_SWITCH, Permission.MANAGE_USERS):
        assert not has_permission(Role.AUDITOR, write)


def test_an_analyst_cannot_record_a_disposition():
    assert not has_permission(Role.ANALYST, Permission.RECORD_DISPOSITION)


def test_require_raises_for_a_role_that_lacks_the_permission():
    with pytest.raises(AccessDenied):
        require(Role.ANALYST, Permission.RECORD_DISPOSITION)
    require(Role.CLAIMS_REVIEWER, Permission.RECORD_DISPOSITION)  # does not raise


# ------------------------------------------------------------ override is distinct


def test_override_is_granted_individually_not_by_seniority():
    """§13: "Override as a DISTINCT permission, not an implicit consequence of seniority"."""
    for role in ALL_ROLES:
        assert not has_permission(role, Permission.MANUAL_OVERRIDE), (
            f"{role.value} holds MANUAL_OVERRIDE by virtue of its role."
        )


def test_admin_does_not_hold_override_by_default():
    """Even the role that holds everything else does not hold this one."""
    assert Permission.MANUAL_OVERRIDE not in ROLE_PERMISSIONS[Role.ADMIN]
    assert ROLE_PERMISSIONS[Role.ADMIN] == set(Permission) - {Permission.MANUAL_OVERRIDE}


def test_override_can_be_granted_to_an_individual():
    assert has_permission(Role.CLAIMS_REVIEWER, Permission.MANUAL_OVERRIDE,
                          [Permission.MANUAL_OVERRIDE.value])


# ------------------------------------------------------- admin-only operations


@pytest.mark.parametrize("permission", [Permission.KILL_SWITCH, Permission.ROLLBACK,
                                        Permission.MANAGE_USERS])
def test_the_dangerous_operations_are_admin_only(permission):
    holders = [r for r in ALL_ROLES if has_permission(r, permission)]
    assert holders == [Role.ADMIN], f"{permission.value} is held by {[r.value for r in holders]}"


def test_model_promotion_needs_two_different_roles():
    """Propose and approve are separate permissions, so they can be separate people."""
    proposers = {r for r in ALL_ROLES if has_permission(r, Permission.PROPOSE_MODEL_PROMOTION)}
    approvers = {r for r in ALL_ROLES if has_permission(r, Permission.APPROVE_MODEL_PROMOTION)}
    assert proposers and approvers
    assert proposers != approvers


# ---------------------------------------------------- separation of duties


def test_an_actor_may_not_approve_their_own_work():
    with pytest.raises(SeparationOfDutiesError) as exc:
        assert_separation_of_duties("priya", "priya", "activate")
    assert "separation of duties" in str(exc.value).lower()


def test_a_different_actor_may_approve():
    assert_separation_of_duties("amina", "priya", "activate")


def test_no_originator_means_no_conflict():
    """A catalogue control loaded from YAML has no author to conflict with."""
    assert_separation_of_duties("amina", None, "activate")
    assert_separation_of_duties("amina", "", "activate")


# ----------------------------------------------------------------- masking


def test_identity_is_masked_by_default_for_a_reviewer():
    masked = mask_identity("H0141", Role.CLAIMS_REVIEWER)
    assert masked != "H0141"
    assert masked.startswith("PROV-")


def test_the_mask_is_a_stable_pseudonym_not_a_blackout():
    """§13.2: mask IDENTITY, not EVIDENCE — a reviewer must still tell them apart."""
    first = mask_identity("H0141", Role.CLAIMS_REVIEWER)
    again = mask_identity("H0141", Role.CLAIMS_REVIEWER)
    other = mask_identity("H0142", Role.CLAIMS_REVIEWER)

    assert first == again, "The same provider must mask to the same pseudonym on every page."
    assert first != other, "Two providers masked to one pseudonym is evidence destroyed."


def test_the_pseudonym_does_not_leak_the_identifier():
    masked = mask_identity("H0141", Role.CLAIMS_REVIEWER)
    assert "H0141" not in masked
    assert "0141" not in masked


def test_members_and_providers_are_visibly_different_kinds_of_subject():
    assert mask_identity("H0141", Role.CLAIMS_REVIEWER).startswith("PROV-")
    assert mask_identity("P00021", Role.CLAIMS_REVIEWER).startswith("MBR-")


def test_admin_sees_identity():
    assert mask_identity("H0141", Role.ADMIN) == "H0141"


def test_an_siu_investigator_sees_identity_on_assigned_cases_only():
    assert mask_identity("H0141", Role.SIU_INVESTIGATOR) != "H0141"
    assert mask_identity("H0141", Role.SIU_INVESTIGATOR, assigned=True) == "H0141"


def test_an_explicit_unmask_grant_reveals_identity():
    assert mask_identity("H0141", Role.CLAIMS_REVIEWER, unmask_granted=True) == "H0141"


def test_a_missing_value_masks_to_a_dash_not_to_a_pseudonym():
    """A pseudonym for a value that does not exist would invent a subject."""
    assert mask_identity(None, Role.CLAIMS_REVIEWER) == "—"


@pytest.mark.parametrize("role", [r for r in ALL_ROLES if r is not Role.ADMIN])
def test_no_role_except_admin_sees_identity_unassigned_and_ungranted(role):
    assert mask_identity("H0141", role) != "H0141"


def test_a_frame_masks_the_named_columns_and_leaves_the_evidence_alone(claims):
    frame = claims.head(20)[["claim_sk", "provider_sk", "member_sk", "gross_amount_aed"]].copy()
    masked = mask_frame(frame, ["provider_sk", "member_sk"], Role.CLAIMS_REVIEWER)

    assert len(masked) == len(frame)
    assert not masked["provider_sk"].isin(frame["provider_sk"]).any()
    assert not masked["member_sk"].isin(frame["member_sk"]).any()
    # The evidence is untouched: masking identity must not degrade the finding.
    assert masked["gross_amount_aed"].tolist() == frame["gross_amount_aed"].tolist()
    assert masked["claim_sk"].tolist() == frame["claim_sk"].tolist()


def test_masking_preserves_the_ability_to_group(claims):
    """The analytic property that a blackout would destroy."""
    frame = claims.head(200)[["provider_sk", "gross_amount_aed"]].copy()
    masked = mask_frame(frame, ["provider_sk"], Role.CLAIMS_REVIEWER)
    assert masked["provider_sk"].nunique() == frame["provider_sk"].nunique()
