Require Import HOLLight_Real_With_N.logic BinNat Lia Coq.Logic.ClassicalEpsilon.

Open Scope N_scope.

Definition N' := {| type := N; el := N0 |}.
Canonical N'.

Lemma N0_or_succ n : n = 0 \/ exists p, n = N.succ p.
Proof. destruct (classic (n = 0)). lia. right. exists (N.pred n). lia. Qed.

Lemma recursion_succ A (a:A) f n :
  N.recursion a f (N.succ n) = f n (N.recursion a f n).
Proof.
  apply N.recursion_succ. reflexivity.
  intros n1 n2 n12 a1 a2 a12. subst n2. subst a2. reflexivity.
Qed.

(****************************************************************************)
(* Alignment of the type of natural numbers. *)
(****************************************************************************)

Definition dest_num := N.recursion IND_0 (fun _ r => IND_SUC r).

Lemma dest_num0 : dest_num 0 = IND_0.
Proof. unfold dest_num. apply N.recursion_0. Qed.

Lemma dest_numS n : dest_num (N.succ n) = IND_SUC (dest_num n).
Proof. unfold dest_num at 1. apply recursion_succ. Qed.

Lemma dest_num_inj : ONE_ONE dest_num.
Proof.
  intro x. pattern x. revert x. apply N.peano_ind.
  intro y. destruct (N0_or_succ y) as [h|[p h]]; subst y. reflexivity.
  rewrite dest_numS.
intro e. apply False_ind. eapply IND_SUC_neq_0. symmetry. exact e.
  intros x hx y. destruct (N0_or_succ y) as [h|[p h]]; subst y.
  rewrite dest_numS.
  intro e. apply False_ind. eapply IND_SUC_neq_0. exact e.
  rewrite !dest_numS. intro e. apply (f_equal N.succ). apply hx.
  apply IND_SUC_inj. exact e.
Qed.

Definition dest_num_img i := exists n, i = dest_num n.

Lemma NUM_REP_eq_dest_num_img : NUM_REP = dest_num_img.
Proof.
  apply fun_ext; intro i. apply prop_ext.
  rewrite NUM_REP_eq_ID. revert i. induction 1.
    exists 0. reflexivity.
    destruct IHNUM_REP_ID as [n hn]. rewrite hn.
    exists (N.succ n). rewrite dest_numS. reflexivity.
  intros [n hn]. subst. pattern n. revert n. apply N.peano_ind.
  rewrite dest_num0. apply NUM_REP_0.
  intros n hn. rewrite dest_numS. apply NUM_REP_S. exact hn.
Qed.

Lemma NUM_REP_dest_num k : NUM_REP (dest_num k).
Proof. rewrite NUM_REP_eq_dest_num_img. exists k. reflexivity. Qed.

Definition mk_num_pred i n := i = dest_num n.

Definition mk_num i := ε (mk_num_pred i).

Lemma mk_num_ex i : NUM_REP i -> exists n, mk_num_pred i n.
Proof.
  rewrite NUM_REP_eq_ID. induction 1.
  exists 0. reflexivity.
  destruct IHNUM_REP_ID as [n hn]. exists (N.succ n). unfold mk_num_pred.
  rewrite hn, dest_numS. reflexivity.
Qed.

Lemma mk_num_prop i : NUM_REP i -> dest_num (mk_num i) = i.
Proof. intro hi. symmetry. apply (ε_spec (mk_num_ex i hi)). Qed.

Notation dest_num_mk_num := mk_num_prop.

Lemma mk_num_dest_num k : mk_num (dest_num k) = k.
Proof. apply dest_num_inj. apply dest_num_mk_num. apply NUM_REP_dest_num. Qed.

Lemma axiom_7 : forall (a : N), (mk_num (dest_num a)) = a.
Proof. exact mk_num_dest_num. Qed.

Lemma axiom_8 : forall (r : ind), (NUM_REP r) = ((dest_num (mk_num r)) = r).
Proof.
  intro r. apply prop_ext. apply dest_num_mk_num. intro h. rewrite <- h.
  apply NUM_REP_dest_num.
Qed.

Lemma mk_num_0 : mk_num IND_0 = 0.
Proof.
  unfold mk_num. set (P := mk_num_pred IND_0).
  assert (h: exists n, P n). exists 0. reflexivity.
  generalize (ε_spec h). set (i := ε P). unfold P, mk_num_pred. intro e.
  apply dest_num_inj. simpl. symmetry. exact e.
Qed.

Lemma _0_def : 0 = (mk_num IND_0).
Proof. symmetry. exact mk_num_0. Qed.

Lemma mk_num_S : forall i, NUM_REP i -> mk_num (IND_SUC i) = N.succ (mk_num i).
Proof.
  intros i hi. rewrite NUM_REP_eq_dest_num_img in hi. destruct hi as [n hn].
  subst i. rewrite mk_num_dest_num, <- dest_numS, mk_num_dest_num. reflexivity.
Qed.

Lemma SUC_def : N.succ = (fun _2104 : N => mk_num (IND_SUC (dest_num _2104))).
Proof.
  symmetry. apply fun_ext; intro x. rewrite mk_num_S. 2: apply NUM_REP_dest_num.
  apply f_equal. apply axiom_7.
Qed.

(****************************************************************************)
(* Alignment of mathematical functions on natural numbers with N. *)
(****************************************************************************)

Definition NUMERAL (x : N) := x.

Definition BIT0 := N.double.

Lemma BIT0_def : BIT0 =
         (@ε (arr N N')
            (fun fn : forall _ : N, N =>
               and (@Logic.eq N (fn N0) N0)
                 (forall n : N, @Logic.eq N (fn (N.succ n)) (N.succ (N.succ (fn n)))))).
Proof.
  match goal with [|- _ = ε ?x] => set (Q := x) end.
  assert (i : exists q, Q q). exists N.double. split. reflexivity. lia.
  generalize (ε_spec i). intros [h0 hs].
  apply fun_ext. apply N.peano_ind.
  rewrite h0. reflexivity. intros n IHn. rewrite hs. unfold BIT0 in *. lia.
Qed.

Definition BIT1 := fun n : N => N.succ (BIT0 n).

Lemma BIT1_eq_succ_double : BIT1 = N.succ_double.
Proof. apply fun_ext; intro n. unfold BIT1, BIT0. lia. Qed.

Lemma PRE_def : N.pred = (@ε (arr (prod N (prod N N)) (arr N N')) (fun PRE' : (prod N (prod N N)) -> N -> N' => forall _2151 : prod N (prod N N), ((PRE' _2151 ( N0)) = ( N0)) /\ (forall n : N, (PRE' _2151 (N.succ n)) = n)) (@pair N (prod N N) ( (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N N ( (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) ( (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))))).
Proof.
  generalize (@pair N (prod N N) ( (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N N ( (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) ( (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))))).
  generalize (prod N (prod N N)).
  intros A a.
  match goal with [|- _ = ε ?x _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => N.pred). split. reflexivity. lia.
  generalize (ε_spec i a). intros [h0 hs].
  apply fun_ext. apply N.peano_ind.
  rewrite h0. reflexivity. intros n IHn. rewrite hs. lia.
Qed.

Lemma add_def : N.add = (@ε (arr N (arr N (arr N N'))) (fun add' : N -> N -> N -> N => forall _2155 : N, (forall n : N, (add' _2155 ( N0) n) = n) /\ (forall m : N, forall n : N, (add' _2155 (N.succ m) n) = (N.succ (add' _2155 m n)))) ( (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))).
Proof.
  generalize ( (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))). intro a.
  match goal with [|- _ = ε ?x _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => N.add). split. reflexivity. lia.
  generalize (ε_spec i a). intros [h0 hs].
  apply fun_ext; intro x. apply fun_ext; intro y.
  revert y. pattern x. revert x. apply N.peano_ind.
  intro y. rewrite h0. lia. intros x IHx y. rewrite hs, N.add_succ_l, IHx.
  reflexivity.
Qed.

Lemma mul_def : N.mul = (@ε (arr N (arr N (arr N N'))) (fun mul' : N -> N -> N -> N => forall _2186 : N, (forall n : N, (mul' _2186 ( N0) n) = ( N0)) /\ (forall m : N, forall n : N, (mul' _2186 (N.succ m) n) = (N.add (mul' _2186 m n) n))) ( (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))).
Proof.
  generalize ( (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))). intro a.
  match goal with [|- _ = ε ?x _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => N.mul). split. reflexivity.
  intros m n. rewrite N.mul_succ_l, N.add_comm. reflexivity.
  generalize (ε_spec i a). intros [h0 hs].
  apply fun_ext; intro x. apply fun_ext; intro y.
  revert y. pattern x. revert x. apply N.peano_ind.
  intro y. rewrite h0. reflexivity.
  intros x IHx y. rewrite N.mul_succ_l, hs, IHx, N.add_comm. reflexivity.
Qed.

Lemma EXP_def : N.pow = (@ε (arr (prod N (prod N N)) (arr N (arr N N'))) (fun EXP' : (prod N (prod N N)) -> N -> N -> N => forall _2224 : prod N (prod N N), (forall m : N, EXP' _2224 m N0 = BIT1 0) /\ (forall m : N, forall n : N, (EXP' _2224 m (N.succ n)) = (N.mul m (EXP' _2224 m n)))) (@pair N (prod N N) (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))) (@pair N N (BIT0 (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 0))))))) (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))).
Proof.
  generalize (@pair N (prod N N) (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))) (@pair N N (BIT0 (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 0))))))) (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))); generalize (@prod N (prod N N)); intros A a.
  match goal with [|- _ = ε ?x _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => N.pow). split. reflexivity.
  intros m n. rewrite N.pow_succ_r. reflexivity. lia.
  generalize (ε_spec i a). intros [h0 hs].
  apply fun_ext; intro x. apply fun_ext. apply N.peano_ind.
  rewrite h0. reflexivity. intros y IHy. rewrite N.pow_succ_r, hs, IHy.
  reflexivity. lia.
Qed.

Lemma le_def : N.le = (@ε (arr (prod N N) (arr N (arr N Prop'))) (fun le' : (prod N N) -> N -> N -> Prop => forall _2241 : prod N N, (forall m : N, (le' _2241 m ( N0)) = (m = ( N0))) /\ (forall m : N, forall n : N, (le' _2241 m (N.succ n)) = ((m = (N.succ n)) \/ (le' _2241 m n)))) (@pair N N ( (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 (BIT1 0))))))) ( (BIT1 (BIT0 (BIT1 (BIT1 (BIT1 (BIT1 0))))))))).
Proof.
  generalize (@pair N N ( (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 (BIT1 0))))))) ( (BIT1 (BIT0 (BIT1 (BIT1 (BIT1 (BIT1 0)))))))); generalize (prod N N); intros A a.
  match goal with [|- _ = ε ?x _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => N.le). split; intros; apply prop_ext; lia.
  generalize (ε_spec i a). intros [h0 hs].
  apply fun_ext; intro x. apply fun_ext; intro y. apply prop_ext.
  revert y. pattern x. revert x. apply N.peano_ind.
    intro y. pattern y. revert y. apply N.peano_ind.
      rewrite h0. reflexivity.
      intros y hy h. rewrite hs. right. apply hy. lia.
    intros x hx y. pattern y. revert y. apply N.peano_ind.
      intro h. lia.
      intros y hy h. rewrite hs. destruct (N.eq_dec x y).
        subst y. left. reflexivity.
        right. apply hy. lia.
  pattern y. revert y. apply N.peano_ind.    
    rewrite h0. lia.
    intros y hy. rewrite hs. intros [h|h].
      lia. transitivity y. apply hy. exact h. lia.
Qed.

Lemma lt_def : N.lt = (@ε (arr N (arr N (arr N Prop'))) (fun lt : N -> N -> N -> Prop => forall _2248 : N, (forall m : N, (lt _2248 m ( N0)) = False) /\ (forall m : N, forall n : N, (lt _2248 m (N.succ n)) = ((m = n) \/ (lt _2248 m n)))) ( (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 (BIT1 0)))))))).
Proof.
  generalize ( (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 (BIT1 0))))))); intro a.
  match goal with [|- _ = ε ?x _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => N.lt). split; intros; apply prop_ext; lia.
  generalize (ε_spec i a). intros [h0 hs].
  apply fun_ext; intro x. apply fun_ext; intro y. apply prop_ext.
  pattern y. revert y. apply N.peano_ind. lia.
  intros y hy h. rewrite hs. destruct (N.eq_dec x y).
  auto. right. apply hy. lia.
  pattern y. revert y. apply N.peano_ind. rewrite h0. lia.
  intros y hy h. rewrite hs in h. destruct h as [h|h]. lia.
  transitivity y. apply hy. exact h. lia.
Qed.

Lemma ge_def : N.ge = (fun _2249 : N => fun _2250 : N => N.le _2250 _2249).
Proof. apply fun_ext; intro x. apply fun_ext; intro y. apply prop_ext; lia. Qed.

Lemma gt_def : N.gt = (fun _2261 : N => fun _2262 : N => N.lt _2262 _2261).
Proof. apply fun_ext; intro x. apply fun_ext; intro y. apply prop_ext; lia. Qed.

Lemma N0_le_eq_True y : N.le 0 y = True.
Proof. apply prop_ext; lia. Qed.

Lemma succ_le_0_is_False x : N.le (N.succ x) 0 = False.
Proof. apply prop_ext; lia. Qed.

Lemma succ_eq_0_is_False x : (N.succ x = N0) = False.
Proof. apply prop_ext; lia. Qed.

Lemma le_succ_succ x y : N.le (N.succ x) (N.succ y) = N.le x y.
Proof. apply prop_ext; lia. Qed.

Lemma MAX_def : N.max = (fun _2273 : N => fun _2274 : N => @COND N (N.le _2273 _2274) _2274 _2273).
Proof.
  apply fun_ext; intro x. apply fun_ext. pattern x. revert x. apply N.peano_ind.
  intro y. rewrite N.max_0_l, N0_le_eq_True, COND_True. reflexivity.
  intros x hx. intro y. pattern y. revert y. apply N.peano_ind.
  rewrite N.max_0_r, succ_le_0_is_False, COND_False. reflexivity.
  intros y hy. rewrite <- N.succ_max_distr, hx, le_succ_succ.
  destruct (prop_degen (N.le x y)) as [h|h]; rewrite h.
  rewrite! COND_True. reflexivity. rewrite! COND_False. reflexivity.
Qed.

Lemma MIN_def : N.min = (fun _2285 : N => fun _2286 : N => @COND N (N.le _2285 _2286) _2285 _2286).
Proof.
  apply fun_ext; intro x. apply fun_ext. pattern x. revert x. apply N.peano_ind.
  intro y. rewrite N.min_0_l, N0_le_eq_True, COND_True. reflexivity.
  intros x hx. intro y. pattern y. revert y. apply N.peano_ind.
  rewrite N.min_0_r, succ_le_0_is_False, COND_False. reflexivity.
  intros y hy. rewrite <- N.succ_min_distr, hx, le_succ_succ.
  destruct (prop_degen (N.le x y)) as [h|h]; rewrite h.
  rewrite! COND_True. reflexivity. rewrite! COND_False. reflexivity.
Qed.

Lemma minus_def : N.sub = (@ε (arr N (arr N (arr N N'))) (fun pair' : N -> N -> N -> N => forall _2766 : N, (forall m : N, (pair' _2766 m ( N0)) = m) /\ (forall m : N, forall n : N, (pair' _2766 m (N.succ n)) = (N.pred (pair' _2766 m n)))) ( (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 0)))))))).
Proof.
  generalize ( (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 0))))))); intro a.
  match goal with [|- _ = ε ?x _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => N.sub). split; lia.
  generalize (ε_spec i a). intros [h0 hs].
  apply fun_ext; intro x. apply fun_ext. pattern x. revert x. apply N.peano_ind.
  intro y. pattern y. revert y. apply N.peano_ind.
  rewrite h0. lia. intros y hy. rewrite hs, <- hy. lia.
  intros x hx y. pattern y. revert y. apply N.peano_ind.
  rewrite h0. lia. intros y hy. rewrite hs, <- hy. lia.
Qed.

(*Lemma FACT_def : Factorial.fact = (@ε (arr (prod nat (prod nat (prod nat nat))) (arr nat nat')) (fun FACT' : (prod nat (prod nat (prod nat nat))) -> nat -> nat => forall _2944 : prod nat (prod nat (prod nat nat)), ((FACT' _2944 ( 0)) = ( (BIT1 0))) /\ (forall n : nat, (FACT' _2944 (S n)) = (Nat.mul (S n) (FACT' _2944 n)))) (@pair nat (prod nat (prod nat nat)) ( (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair nat (prod nat nat) ( (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair nat nat ( (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) ( (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))))).
Proof.
  generalize (@pair nat (prod nat (prod nat nat)) ( (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair nat (prod nat nat) ( (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair nat nat ( (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) ( (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))); generalize (prod nat (prod nat (prod nat nat))); intros A a.
  match goal with [|- _ = ε ?x _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => Factorial.fact). split; reflexivity.
  generalize (ε_spec i a). intros [h0 hs].
  apply fun_ext; intro x. induction x. rewrite h0. reflexivity. rewrite hs, <- IHx. reflexivity.
Qed.*)

Lemma Nadd_sub a b : a + b - a = b. Proof. lia. Qed.

Lemma Nswap_add_sub a a' b : a' <= a -> a + b - a' = a - a' + b. Proof. lia. Qed.

Lemma Ndivmod_unicity k k' q r r' :
  r < q -> r' < q -> k * q + r = k' * q + r' -> k = k' /\ r = r'.
Proof.
  intros h h' e. destruct (classic (N.lt k k')).
  apply False_rec.
  assert (e2 : k * q + r - k' * q = k' * q + r' - k' * q). lia.
  rewrite Nadd_sub, Nswap_add_sub, <- N.mul_sub_distr_r in e2. nia. nia.
  destruct (classic (N.lt k' k)).
  assert (e2 : k * q + r - k' * q = k' * q + r' - k' * q). lia.
  rewrite Nadd_sub, Nswap_add_sub, <- N.mul_sub_distr_r in e2. nia. nia.
  nia.
Qed.

Lemma DIV_def : N.div = (@ε (arr (prod N (prod N N)) (arr N (arr N N'))) (fun q : (prod N (prod N N)) -> N -> N -> N => forall _3086 : prod N (prod N N), exists r : N -> N -> N, forall m : N, forall n : N, @COND Prop (n = ( N0)) (((q _3086 m n) = ( N0)) /\ ((r m n) = m)) ((m = (N.add (N.mul (q _3086 m n) n) (r m n))) /\ (N.lt (r m n) n))) (@pair N (prod N N) ( (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N ( (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) ( (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))).
Proof.
  generalize (@pair N (prod N N) ( (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N ( (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) ( (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))); generalize (prod N (prod N (prod N N))); intros A a.
  match goal with [|- _ = ε ?x _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => N.div). intro x. exists N.modulo. intros m n.
  destruct (prop_degen (n=N0)) as [h|h]; rewrite h.
  rewrite COND_True. rewrite is_True in h. subst n. split.
  apply N.div_0_r. apply N.mod_0_r.
  rewrite COND_False, N.mul_comm, <- N.div_mod'. split. reflexivity.
  apply N.mod_lt. rewrite <- is_False. exact h.

  generalize (ε_spec i a). intros [mod' h].
  apply fun_ext; intro x. apply fun_ext; intro y.
  revert x. pattern y. revert y. apply N.peano_ind.

  intro x. generalize (h x N0). rewrite refl_is_True, COND_True. intros [h1 h2].
  rewrite h1. apply N.div_0_r.

  intros y hy x. generalize (h x (N.succ y)).
  rewrite succ_eq_0_is_False, COND_False. intros [h1 h2].
  generalize (N.div_mod' x (N.succ y)). rewrite N.mul_comm, h1 at 1. intro h3.
  apply Ndivmod_unicity in h3. destruct h3 as [h3 h4]. auto.
  exact h2. apply N.mod_lt. lia.
Qed.

Lemma MOD_def : N.modulo = (@ε (arr (prod N (prod N N)) (arr N (arr N N'))) (fun r : (prod N (prod N N)) -> N -> N -> N => forall _3087 : prod N (prod N N), forall m : N, forall n : N, @COND Prop (n = ( 0)) (((N.div m n) = ( 0)) /\ ((r _3087 m n) = m)) ((m = (N.add (N.mul (N.div m n) n) (r _3087 m n))) /\ (N.lt (r _3087 m n) n))) (@pair N (prod N N) ( (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N ( (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) ( (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))))).
Proof.
  generalize (@pair N (prod N N) ( (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N ( (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) ( (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))))); generalize (prod N (prod N (prod N N))); intros A a.
  match goal with [|- _ = ε ?x _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => N.modulo). intros x m n.
  destruct (prop_degen (n=N0)) as [h|h]; rewrite h.
  rewrite COND_True. rewrite is_True in h. subst n. split.
  apply N.div_0_r. apply N.mod_0_r.
  rewrite COND_False, N.mul_comm, <- N.div_mod'. split. reflexivity.
  apply N.mod_lt. rewrite <- is_False. exact h.

  generalize (ε_spec i a); intro h.
  apply fun_ext; intro x. apply fun_ext; intro y.
  revert x. pattern y. revert y. apply N.peano_ind.

  intro x. generalize (h x N0). rewrite refl_is_True, COND_True. intros [h1 h2].
  rewrite N.mod_0_r. auto.

  intros y hy x. generalize (h x (N.succ y)).
  rewrite succ_eq_0_is_False, COND_False. intros [h1 h2].
  generalize (N.div_mod' x (N.succ y)). rewrite N.mul_comm, h1 at 1. intro h3.
  apply Ndivmod_unicity in h3. destruct h3 as [h3 h4]. auto.
  exact h2. apply N.mod_lt. lia.
Qed.

(****************************************************************************)
(* Alignment of the Even and Odd predicates. *)
(****************************************************************************)

(*Import PeanoNat.Nat Private_Parity.

Lemma odd_double n : odd (2 * n) = false.
Proof. rewrite odd_mul, odd_2. reflexivity. Qed.

Lemma even_double n : even (2 * n) = true.
Proof. rewrite even_spec. exists n. reflexivity. Qed.

Lemma Even_or_Odd x : Even x \/ Odd x.
Proof.
  rewrite (div_mod_eq x 2). assert (h1: 0 <= x). lia. assert (h2: 0 < 2). lia.
  generalize (mod_bound_pos x 2 h1 h2). generalize (x mod 2). intro n.
  destruct n; intro h.
  left. exists (x / 2). lia. destruct n. right. exists (x / 2). reflexivity. lia.
Qed.

Lemma not_Even_is_Odd x : (~Even x) = Odd x.
Proof.
  apply prop_ext; intro h; generalize (Even_or_Odd x); intros [i|i].
  apply False_rec. exact (h i). exact i. destruct h as [k hk].
  destruct i as [l hl]. lia.
  intros [k hk]. destruct i as [l hl]. lia.
Qed.

Lemma not_Odd_is_Even x : (~Odd x) = Even x.
Proof.
  apply prop_ext; intro h; generalize (Even_or_Odd x); intros [i|i].
  exact i. apply False_rec. exact (h i). destruct h as [k hk]. intro j.
  destruct j as [l hl]. lia.
  intros [k hk]. destruct h as [l hl]. lia.
Qed.

Lemma Even_S x : Even (S x) = Odd x.
Proof.
  apply prop_ext; intros [k hk].
  rewrite <- not_Even_is_Odd. intros [l hl]. lia.
  rewrite <- not_Odd_is_Even. intros [l hl]. lia.
Qed.

Lemma Odd_S x : Odd (S x) = Even x.
Proof.
  apply prop_ext; intros [k hk].
  rewrite <- not_Odd_is_Even. intros [l hl]. lia.
  rewrite <- not_Even_is_Odd. intros [l hl]. lia.
Qed.

Lemma EVEN_def : Even = (@ε (arr (prod nat (prod nat (prod nat nat))) (arr nat Prop')) (fun EVEN' : (prod nat (prod nat (prod nat nat))) -> nat -> Prop => forall _2603 : prod nat (prod nat (prod nat nat)), ((EVEN' _2603 (0)) = True) /\ (forall n : nat, (EVEN' _2603 (S n)) = (~ (EVEN' _2603 n)))) (@pair nat (prod nat (prod nat nat)) ((BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair nat (prod nat nat) ((BIT0 (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair nat nat ((BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) ((BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))))))).
Proof.
  generalize (@pair nat (prod nat (prod nat nat)) ((BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair nat (prod nat nat) ((BIT0 (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair nat nat ((BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) ((BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))))))); generalize (prod nat (prod nat (prod nat nat))); intros A a.
  match goal with [|- _ = ε ?x _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => Even). intro x. split.
  rewrite is_True. exact Even_0. intro n. rewrite not_Even_is_Odd. apply Even_S.
  generalize (ε_spec i a). intros [h1 h2].
  apply fun_ext; intro x. induction x.
  apply prop_ext; intro h. rewrite h1. exact I. exact Even_0.
  rewrite h2, <- IHx, not_Even_is_Odd. apply Even_S.
Qed.

Lemma ODD_def : Odd = (@ε (arr (prod nat (prod nat nat)) (arr nat Prop')) (fun ODD' : (prod nat (prod nat nat)) -> nat -> Prop => forall _2607 : prod nat (prod nat nat), ((ODD' _2607 (0)) = False) /\ (forall n : nat, (ODD' _2607 (S n)) = (~ (ODD' _2607 n)))) (@pair nat (prod nat nat) ((BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair nat nat ((BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) ((BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))))).
Proof.
  generalize (@pair nat (prod nat nat) ((BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair nat nat ((BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) ((BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))))); generalize (prod nat (prod nat (prod nat nat))); intros A a.
  match goal with [|- _ = ε ?x _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => Odd). intro x. split.
  rewrite is_False. exact Odd_0. intro n. rewrite not_Odd_is_Even. apply Odd_S.
  generalize (ε_spec i a). intros [h1 h2].
  apply fun_ext; intro x. induction x.
  apply prop_ext; intro h. rewrite h1. apply Odd_0. exact h.
  apply False_rec. rewrite <- h1. exact h.
  rewrite h2, <- IHx, not_Odd_is_Even. apply Odd_S.
Qed.*)

(****************************************************************************)
(* NUMPAIR(x,y) = 2^x(2y+1): bijection between N² and N-{0}. *)
(****************************************************************************)

Definition NUMPAIR := fun x : N => fun y : N => N.mul (N.pow (NUMERAL (BIT0 (BIT1 0))) x) (N.add (N.mul (NUMERAL (BIT0 (BIT1 0))) y) (NUMERAL (BIT1 0))).

Lemma double_0 : N.double 0 = 0. Proof. lia. Qed.

Lemma succ_0 : N.succ 0 = 1. Proof. lia. Qed.

Lemma double_1 : N.double 1 = 2. Proof. lia. Qed.

Lemma lt2le {a b} : (a < b) = (N.succ a <= b).
Proof. apply prop_ext; lia. Qed.

Lemma le_is_add {a b} : a <= b -> exists c, b = a + c.
Proof. intro h. exists (b - a). lia. Qed.

Lemma eq_mul_r {a b} : a <> 0 -> a = b * a -> b = 1.
Proof.
  intro h. rewrite <- (N.mul_1_l a) at 1. intro e. apply Nmult_reg_r in e.
  auto. auto.
Qed.

Lemma NDIV_MULT m n : m <> 0 -> (m * n) / m = n.
Proof. intro h. rewrite N.mul_comm. apply N.div_mul. exact h. Qed.

Lemma NUMPAIR_INJ : forall x1 : N, forall y1 : N, forall x2 : N, forall y2 : N, ((NUMPAIR x1 y1) = (NUMPAIR x2 y2)) = ((x1 = x2) /\ (y1 = y2)).
Proof.
  intros x1 y1 x2 y2. apply prop_ext. 2: intros [e1 e2]; subst; reflexivity.
  unfold NUMPAIR, NUMERAL, BIT1, BIT0. rewrite double_0, succ_0, double_1.
  intro e.

  destruct (classic (x1 < x2)) as [h|h]. rewrite lt2le in h.
  apply False_rec. destruct (le_is_add h) as [k hk]. subst x2.
  generalize (f_equal (fun x => N.div x (2 ^ x1)) e).
  rewrite NDIV_MULT, N.pow_add_r, N.pow_succ_r, (N.mul_comm 2 (2 ^ x1)),
    <- !N.mul_assoc, NDIV_MULT.
  lia. lia. lia. lia.
  
  destruct (classic (x2 < x1)) as [i|i]. rewrite lt2le in i.
  apply False_rec. destruct (le_is_add i) as [k hk]. subst x1.
  generalize (f_equal (fun x => N.div x (2 ^ x2)) e).
  rewrite NDIV_MULT, N.pow_add_r, N.pow_succ_r, (N.mul_comm 2 (2 ^ x2)),
    <- !N.mul_assoc, NDIV_MULT.
  lia. lia. lia. lia.

  assert (j: x1 = x2). lia. subst x2. split. reflexivity. nia.
Qed.

Lemma NUMPAIR_nonzero x y : NUMPAIR x y <> 0.
Proof.
  unfold NUMPAIR, NUMERAL, BIT1, BIT0.
  rewrite double_0, succ_0, double_1, N.mul_add_distr_l, N.mul_1_r. nia.
Qed.

(****************************************************************************)
(* Inverse of NUMPAIR. *)
(****************************************************************************)

Lemma INJ_INVERSE2 {A B C : Type'} : forall P : A -> B -> C, (forall x1 : A, forall y1 : B, forall x2 : A, forall y2 : B, ((P x1 y1) = (P x2 y2)) = ((x1 = x2) /\ (y1 = y2))) -> exists X : C -> A, exists Y : C -> B, forall x : A, forall y : B, ((X (P x y)) = x) /\ ((Y (P x y)) = y).
Proof.
  intros f h.
  exists (fun c => ε (fun a => exists b, f a b = c)).
  exists (fun c => ε (fun b => exists a, f a b = c)).
  intros a b. split.
  match goal with [|- ε ?x = _] => set (Q := x); set (q := ε Q) end.
  assert (i : exists a, Q a). exists a. exists b. reflexivity.
  generalize (ε_spec i). fold q. unfold Q. intros [b' j]. rewrite h in j.
  destruct j as [j1 j2]. auto.
  match goal with [|- ε ?x = _] => set (Q := x); set (q := ε Q) end.
  assert (i : exists b, Q b). exists b. exists a. reflexivity.
  generalize (ε_spec i). fold q. unfold Q. intros [a' j]. rewrite h in j.
  destruct j as [j1 j2]. auto.
Qed.

Definition NUMFST0_pred :=  fun X : N -> N => exists Y : N -> N, forall x : N, forall y : N, ((X (NUMPAIR x y)) = x) /\ ((Y (NUMPAIR x y)) = y).

Definition NUMFST0 := ε NUMFST0_pred.

Lemma NUMFST0_NUMPAIR x y : NUMFST0 (NUMPAIR x y) = x.
Proof.
  destruct (INJ_INVERSE2 _ NUMPAIR_INJ) as [fst [snd h]].
  assert (i : exists q, NUMFST0_pred q). exists fst. exists snd. assumption.
  generalize (ε_spec i). fold NUMFST0. unfold NUMFST0_pred.
  intros [snd' h']. destruct (h' x y) as [j k]. assumption.
Qed.

Definition NUMSND0_pred :=  fun Y : N -> N => exists X : N -> N, forall x : N, forall y : N, ((X (NUMPAIR x y)) = x) /\ ((Y (NUMPAIR x y)) = y).

Definition NUMSND0 := ε NUMSND0_pred.

Lemma NUMSND0_NUMPAIR x y : NUMSND0 (NUMPAIR x y) = y.
Proof.
  destruct (INJ_INVERSE2 _ NUMPAIR_INJ) as [fst [snd h]].
  assert (i : exists x, NUMSND0_pred x). exists snd. exists fst. assumption.
  generalize (ε_spec i). fold NUMSND0. unfold NUMSND0_pred.
  intros [fst' h']. destruct (h' x y) as [j k]. assumption.
Qed.

Definition NUMFST := @ε ((prod N (prod N (prod N (prod N (prod N N))))) -> N -> N) (fun X : (prod N (prod N (prod N (prod N (prod N N))))) -> N -> N => forall _17340 : prod N (prod N (prod N (prod N (prod N N)))), exists Y : N -> N, forall x : N, forall y : N, ((X _17340 (NUMPAIR x y)) = x) /\ ((Y (NUMPAIR x y)) = y)) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))))).

Lemma NUMFST_NUMPAIR x y : NUMFST (NUMPAIR x y) = x.
Proof.
  unfold NUMFST.
  generalize (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
     (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
      (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
       (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
        (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
          NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))))).
  generalize (prod N (prod N (prod N (prod N (prod N N))))); intros A a.
  match goal with |- ε ?x _ _ = _ => set (Q := x); set (fst := ε Q) end.
  assert (i: exists x, Q x). exists (fun _ => NUMFST0). unfold Q. intros _. exists NUMSND0.
  intros x' y'. rewrite NUMFST0_NUMPAIR, NUMSND0_NUMPAIR. auto.
  generalize (ε_spec i). change (Q fst -> fst a (NUMPAIR x y) = x). intro h.
  destruct (h a) as [snd j]. destruct (j x y) as [j1 j2]. exact j1.
Qed.

Definition NUMSND1_pred :=  fun Y : N -> N => forall x : N, forall y : N, ((NUMFST (NUMPAIR x y)) = x) /\ ((Y (NUMPAIR x y)) = y).

Definition NUMSND1 := ε NUMSND1_pred.

Lemma NUMSND1_NUMPAIR x y : NUMSND1 (NUMPAIR x y) = y.
Proof.
  destruct (INJ_INVERSE2 _ NUMPAIR_INJ) as [fst [snd h]].
  assert (i : exists x, NUMSND1_pred x). exists snd. unfold NUMSND1_pred.
  intros x' y'. rewrite NUMFST_NUMPAIR. destruct (h x' y') as [h1 h2]. auto.
  generalize (ε_spec i). fold NUMSND1. unfold NUMSND1_pred. intro j.
  destruct (j x y) as [j1 j2]. exact j2.
Qed.

Definition NUMSND := @ε ((prod N (prod N (prod N (prod N (prod N N))))) -> N -> N) (fun Y : (prod N (prod N (prod N (prod N (prod N N))))) -> N -> N => forall _17341 : prod N (prod N (prod N (prod N (prod N N)))), forall x : N, forall y : N, ((NUMFST (NUMPAIR x y)) = x) /\ ((Y _17341 (NUMPAIR x y)) = y)) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))))))).

Lemma NUMSND_NUMPAIR x y : NUMSND (NUMPAIR x y) = y.
Proof.
  unfold NUMSND.
  generalize  (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
     (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
      (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
       (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
        (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
         NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))))))).
  generalize (prod N (prod N (prod N (prod N (prod N N))))); intros A a.
  match goal with |- ε ?x _ _ = _ => set (Q := x); set (snd := ε Q) end.
  assert (i: exists x, Q x). exists (fun _ => NUMSND1). unfold Q. intros _.
  intros x' y'. rewrite NUMFST_NUMPAIR, NUMSND1_NUMPAIR. auto.
  generalize (ε_spec i). change (Q snd -> snd a (NUMPAIR x y) = y). intro h.
  destruct (h a x y) as [h1 h2]. exact h2.
Qed.

(****************************************************************************)
(* NUMSUM(b,n) = if b then 2n+1 else 2n : bijection between BxN and N. *)
(****************************************************************************)

Definition NUMSUM := fun b : Prop => fun n : N => @COND N b (N.succ (N.mul (NUMERAL (BIT0 (BIT1 0))) n)) (N.mul (NUMERAL (BIT0 (BIT1 0))) n).

Definition NUMLEFT n := if N.even n then False else True.

Definition NUMRIGHT n := if N.even n then n / 2 else (n - 1) / 2.

Lemma even_double n : N.even (2 * n) = true.
Proof. rewrite N.even_spec. exists n. reflexivity. Qed.

Lemma NUMLEFT_NUMSUM b n : NUMLEFT (NUMSUM b n) = b.
Proof.
  unfold NUMSUM, NUMERAL, BIT1, BIT0, NUMLEFT.
  destruct (prop_degen b); subst; rewrite double_0, succ_0, double_1.
  rewrite COND_True, N.even_succ, N.odd_mul, N.odd_2. reflexivity.
  rewrite COND_False, even_double. reflexivity.
Qed.

Lemma succ_minus_1 x : N.succ x - 1 = x.
Proof. lia. Qed.

Lemma NUMRIGHT_NUMSUM b n : NUMRIGHT (NUMSUM b n) = n.
Proof.
  unfold NUMSUM, NUMERAL, BIT1, BIT0, NUMRIGHT.
  destruct (prop_degen b); subst; rewrite double_0, succ_0, double_1.
  rewrite COND_True, N.even_succ, N.odd_mul, N.odd_2, succ_minus_1, NDIV_MULT.
  reflexivity. lia.
  rewrite COND_False, even_double, NDIV_MULT. reflexivity. lia.
Qed.

Lemma Nplus_1_minus_1 x : x + 1 - 1 = x.
Proof. lia. Qed.

Lemma NUMSUM_surjective n : exists b x, n = NUMSUM b x.
Proof.
  exists (NUMLEFT n). exists (NUMRIGHT n). unfold NUMSUM, NUMLEFT, NUMRIGHT, NUMERAL, BIT1, BIT0.
  case_eq (N.even n); intro h.
  rewrite COND_False. rewrite N.even_spec in h. destruct h as [k h]. subst n.
  rewrite NDIV_MULT. reflexivity. lia.
  rewrite COND_True. apply eq_false_negb_true in h. change (N.odd n = true) in h.
  rewrite N.odd_spec in h. destruct h as [k h]. subst. rewrite Nplus_1_minus_1.
  rewrite NDIV_MULT. lia. lia.
Qed.

Lemma NUMLEFT_def : NUMLEFT = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> N -> Prop) (fun X : (prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> N -> Prop => forall _17372 : prod N (prod N (prod N (prod N (prod N (prod N N))))), exists Y : N -> N, forall x : Prop, forall y : N, ((X _17372 (NUMSUM x y)) = x) /\ ((Y (NUMSUM x y)) = y)) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))))))).
Proof.
  generalize (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
     (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
      (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
       (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
        (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
         (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
           NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))))).
  generalize (prod N (prod N (prod N (prod N (prod N (prod N N)))))); intros A a.
  match goal with |- _ = ε ?x _ => set (Q := x) end.
  assert (i: exists q, Q q). exists (fun _ => NUMLEFT). intros _. exists NUMRIGHT.
  intros b x. rewrite NUMLEFT_NUMSUM, NUMRIGHT_NUMSUM. auto.
  generalize (ε_spec i); intro h. destruct (h a) as [snd j].
  apply fun_ext; intro n. destruct (NUMSUM_surjective n) as [b [x k]]. subst.
  rewrite NUMLEFT_NUMSUM. destruct (j b x) as [j1 j2]. auto.
Qed.

Lemma NUMRIGHT_def : NUMRIGHT = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> N -> N) (fun Y : (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> N -> N => forall _17373 : prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))), forall x : Prop, forall y : N, ((NUMLEFT (NUMSUM x y)) = x) /\ ((Y _17373 (NUMSUM x y)) = y)) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N N)))))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))))))))).
Proof.
  generalize (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
     (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
      (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
       (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
        (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
         (NUMERAL (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
          (NUMERAL (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
           NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))))))).
  generalize (prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))));
    intros A a.
  match goal with |- _ = ε ?x _ => set (Q := x) end.
  assert (i: exists q, Q q). exists (fun _ => NUMRIGHT). intros _ b x.
  rewrite NUMLEFT_NUMSUM, NUMRIGHT_NUMSUM. auto.
  generalize (ε_spec i); intro h.
  apply fun_ext; intro n. destruct (NUMSUM_surjective n) as [b [x k]]. subst.
  rewrite NUMRIGHT_NUMSUM. destruct (h a b x) as [j1 j2]. auto.
Qed.

(****************************************************************************)
(* Alignment of recspace, the HOL-Light type used to encode inductive types. *)
(****************************************************************************)

Definition INJN {A : Type'} := fun x : N => fun n : N => fun a : A => n = x.

Definition INJA {A : Type'} := fun x : A => fun n : N => fun b : A => b = x.

Definition INJF {A : Type'} := fun f : N -> N -> A -> Prop => fun n : N => f (NUMFST n) (NUMSND n).

Definition INJP {A : Type'} := fun f : N -> A -> Prop => fun g : N -> A -> Prop => fun n : N => fun a : A => @COND Prop (NUMLEFT n) (f (NUMRIGHT n) a) (g (NUMRIGHT n) a).

Definition ZCONSTR {A : Type'} := fun n : N => fun a : A => fun f : N -> N -> A -> Prop => @INJP A (@INJN A (N.succ n)) (@INJP A (@INJA A a) (@INJF A f)).

Definition ZBOT {A : Type'} := @INJP A (@INJN A (NUMERAL 0)) (@ε (N -> A -> Prop) (fun z : N -> A -> Prop => True)).

Inductive ZRECSPACE {A : Type'} : (N -> A -> Prop) -> Prop :=
| ZRECSPACE0 : ZRECSPACE ZBOT
| ZRECSPACE1 c i r : (forall n, ZRECSPACE (r n)) -> ZRECSPACE (ZCONSTR c i r).

Lemma ZRECSPACE_def {A : Type'} : (@ZRECSPACE A) = (fun a : N -> A -> Prop => forall ZRECSPACE' : (N -> A -> Prop) -> Prop, (forall a' : N -> A -> Prop, ((a' = (@ZBOT A)) \/ (exists c : N, exists i : A, exists r : N -> N -> A -> Prop, (a' = (@ZCONSTR A c i r)) /\ (forall n : N, ZRECSPACE' (r n)))) -> ZRECSPACE' a') -> ZRECSPACE' a).
Proof.
  apply fun_ext; intro a. apply prop_ext.
  induction 1; intros a h; apply h. left. reflexivity.
  right. exists c. exists i. exists r. split. reflexivity. intro n. apply (H0 n a h).
  intro h. apply h. intros a' [e|[c [i [r [e j]]]]]; subst.
  apply ZRECSPACE0. apply ZRECSPACE1. exact j.
Qed.

Definition recspace : Type' -> Type' := fun A => subtype (@ZRECSPACE0 A).

Definition _dest_rec : forall {A : Type'}, (recspace A) -> N -> A -> Prop :=
  fun A => dest (@ZRECSPACE0 A).

Definition _mk_rec : forall {A : Type'}, (N -> A -> Prop) -> recspace A :=
  fun A => mk (@ZRECSPACE0 A).

Lemma axiom_10 : forall {A : Type'} (r : N -> A -> Prop), (@ZRECSPACE A r) = ((@_dest_rec A (@_mk_rec A r)) = r).
Proof. intros A r. apply dest_mk. Qed.

Lemma axiom_9 : forall {A : Type'} (a : recspace A), (@_mk_rec A (@_dest_rec A a)) = a.
Proof. intros A a. apply mk_dest. Qed.

Definition BOTTOM {A : Type'} := @_mk_rec A (@ZBOT A).

Definition CONSTR {A : Type'} := fun n : N => fun a : A => fun f : N -> recspace A => @_mk_rec A (@ZCONSTR A n a (fun x : N => @_dest_rec A (f x))).

Lemma NUMSUM_INJ : forall b1 : Prop, forall x1 : N, forall b2 : Prop, forall x2 : N, ((NUMSUM b1 x1) = (NUMSUM b2 x2)) = ((b1 = b2) /\ (x1 = x2)).
Proof.
  intros b1 x1 b2 x2. apply prop_ext. 2: intros [e1 e2]; subst; reflexivity.
  unfold NUMSUM. unfold NUMERAL, BIT1, BIT0.
  destruct (prop_degen b1); destruct (prop_degen b2); subst; try rewrite !COND_True; try rewrite !COND_False; intro e.
  split. auto. lia.
  apply False_rec. lia.
  apply False_rec. lia.
  split. auto. lia.
Qed.

Lemma INJN_INJ {A : Type'} : forall n1 : N, forall n2 : N, ((@INJN A n1) = (@INJN A n2)) = (n1 = n2).
Proof.
  intros n1 n2. apply prop_ext. 2: intro e; subst; reflexivity.
  intro e. generalize (ext_fun e n1); clear e; intro e.
  generalize (ext_fun e (el A)); clear e. unfold INJN.
  rewrite refl_is_True, sym, is_True. auto.
Qed.

Lemma INJA_INJ {A : Type'} : forall a1 : A, forall a2 : A, ((@INJA A a1) = (@INJA A a2)) = (a1 = a2).
Proof.
  intros a1 a2. apply prop_ext. 2: intro e; subst; reflexivity.
  intro e. generalize (ext_fun e 0); clear e; intro e.
  generalize (ext_fun e a2); clear e. unfold INJA.
  rewrite refl_is_True, is_True. auto.
Qed.

Lemma INJF_INJ {A : Type'} : forall f1 : N -> N -> A -> Prop, forall f2 : N -> N -> A -> Prop, ((@INJF A f1) = (@INJF A f2)) = (f1 = f2).
Proof.
  intros f1 f2. apply prop_ext. 2: intro e; subst; reflexivity.
  intro e. apply fun_ext; intro x. apply fun_ext; intro y.
  apply fun_ext; intro a.
  generalize (ext_fun e (NUMPAIR x y)); clear e; intro e.
  generalize (ext_fun e a); clear e. unfold INJF.
  rewrite NUMFST_NUMPAIR, NUMSND_NUMPAIR. auto.
Qed.

Lemma Nodd_double n : N.odd (2 * n) = false.
Proof. rewrite N.odd_mul, N.odd_2. reflexivity. Qed.

Lemma Neven_double n : N.even (2 * n) = true.
Proof. rewrite N.even_spec. exists n. reflexivity. Qed.

Lemma INJP_INJ {A : Type'} : forall f1 : N -> A -> Prop, forall f1' : N -> A -> Prop, forall f2 : N -> A -> Prop, forall f2' : N -> A -> Prop, ((@INJP A f1 f2) = (@INJP A f1' f2')) = ((f1 = f1') /\ (f2 = f2')).
Proof.
  intros f1 f1' f2 f2'. apply prop_ext. 2: intros [e1 e2]; subst; reflexivity.
  intro e.
  assert (e1: forall x a, INJP f1 f2 x a = INJP f1' f2' x a).
  intros x a. rewrite e. reflexivity.
  split; apply fun_ext; intro x; apply fun_ext; intro a.
  generalize (e1 (N.succ (2 * x)) a). unfold INJP, NUMLEFT, NUMRIGHT.
  rewrite N.even_succ, Nodd_double, !COND_True, succ_minus_1, NDIV_MULT. auto. lia.
  generalize (e1 (2 * x) a). unfold INJP, NUMLEFT, NUMRIGHT.
  rewrite Neven_double, !COND_False, NDIV_MULT. auto. lia.
Qed.

Lemma ZCONSTR_INJ {A : Type'} c1 i1 r1 c2 i2 r2 : @ZCONSTR A c1 i1 r1 = ZCONSTR c2 i2 r2 -> c1 = c2 /\ i1 = i2 /\ r1 = r2.
Proof.
  unfold ZCONSTR. intro e.
  rewrite INJP_INJ in e. destruct e as [e1 e2].
  rewrite INJN_INJ in e1. rewrite INJP_INJ in e2. destruct e2 as [e2 e3].
  rewrite INJA_INJ in e2. rewrite INJF_INJ in e3. apply N.succ_inj in e1. auto.
Qed.

Lemma MK_REC_INJ {A : Type'} : forall x : N -> A -> Prop, forall y : N -> A -> Prop, ((@_mk_rec A x) = (@_mk_rec A y)) -> ((@ZRECSPACE A x) /\ (@ZRECSPACE A y)) -> x = y.
Proof.
  intros x y e [hx hy]. rewrite axiom_10 in hx. rewrite axiom_10 in hy.
  rewrite <- hx, <- hy, e. reflexivity.
Qed.

Lemma CONSTR_INJ : forall {A : Type'}, forall c1 : N, forall i1 : A, forall r1 : N -> recspace A, forall c2 : N, forall i2 : A, forall r2 : N -> recspace A, ((@CONSTR A c1 i1 r1) = (@CONSTR A c2 i2 r2)) = ((c1 = c2) /\ ((i1 = i2) /\ (r1 = r2))).
Proof.
  intros A c1 i1 r1 c2 i2 r2. apply prop_ext.
  2: intros [e1 [e2 e3]]; subst; reflexivity.
  unfold CONSTR. intro e. apply MK_REC_INJ in e. apply ZCONSTR_INJ in e.
  destruct e as [e1 [e2 e3]]. split. auto. split. auto. apply fun_ext; intro x.
  apply dest_inj. generalize (ext_fun e3 x). auto.
  split; apply ZRECSPACE1; intro n. destruct (r1 n). auto. destruct (r2 n). auto.
Qed.

(****************************************************************************)
(* Alignment of the sum type constructor. *)
(****************************************************************************)

Definition sum' (A B : Type') : Type' := {| type:= sum A B; el := inl (el A) |}.
Canonical sum'.

Definition _dest_sum : forall {A B : Type'}, sum A B -> recspace (prod A B) :=
fun A B p => match p with
| inl a => CONSTR (NUMERAL N0) (a , ε (fun _ => True)) (fun _ => BOTTOM)
| inr b => CONSTR (N.succ (NUMERAL N0)) (ε (fun _ => True) , b) (fun _ => BOTTOM)
end.

Definition _mk_sum : forall {A B : Type'}, recspace (prod A B) -> sum A B :=
  fun A B f => ε (fun p => f = _dest_sum p).

Lemma _dest_sum_inj :
  forall {A B : Type'} (f g : sum A B), _dest_sum f = _dest_sum g -> f = g.
Proof.
  intros.
  induction f; induction g; unfold _dest_sum in H; rewrite (@CONSTR_INJ (prod A B)) in H; destruct H. destruct H0.
  apply pair_equal_spec in H0. destruct H0. rewrite H0. reflexivity.
  discriminate. discriminate.
  destruct H0. apply pair_equal_spec in H0. destruct H0. rewrite H2. reflexivity.
Qed.

Lemma axiom_11 : forall {A B : Type'} (a : sum A B), (@_mk_sum A B (@_dest_sum A B a)) = a.
Proof.
  intros A B a. unfold _mk_sum. apply _dest_sum_inj.
  rewrite sym. apply (@ε_spec (sum A B)). exists a. reflexivity.
Qed.

Lemma axiom_12 : forall {A B : Type'} (r : recspace (prod A B)), ((fun a : recspace (prod A B) => forall sum' : (recspace (prod A B)) -> Prop, (forall a' : recspace (prod A B), ((exists a'' : A, a' = ((fun a''' : A => @CONSTR (prod A B) (NUMERAL 0) (@pair A B a''' (@ε B (fun v : B => True))) (fun n : N => @BOTTOM (prod A B))) a'')) \/ (exists a'' : B, a' = ((fun a''' : B => @CONSTR (prod A B) (N.succ (NUMERAL N0)) (@pair A B (@ε A (fun v : A => True)) a''') (fun n : N => @BOTTOM (prod A B))) a''))) -> sum' a') -> sum' a) r) = ((@_dest_sum A B (@_mk_sum A B r)) = r).
Proof.
  intros. apply prop_ext.
  intro h. unfold _mk_sum. rewrite sym. apply (@ε_spec (sum' A B)).
  apply (h (fun r : recspace (prod A B) => exists x : sum' A B, r = _dest_sum x)).
  intros. destruct H. destruct H. exists (inl(x)). simpl. exact H.

  destruct H. exists (inr(x)). simpl. exact H.

  intro e. rewrite <- e. intros P h. apply h. destruct (_mk_sum r).
  simpl. left. exists t. reflexivity. right. exists t. reflexivity.
Qed.

Lemma INL_def {A B : Type'} : (@inl A B) = (fun a : A => @_mk_sum A B ((fun a' : A => @CONSTR (prod A B) (NUMERAL 0) (@pair A B a' (@ε B (fun v : B => True))) (fun n : N => @BOTTOM (prod A B))) a)).
Proof.
  apply fun_ext. intro a. apply _dest_sum_inj. simpl.
  match goal with [|- ?x = _] => set (r := x) end.
  (* rewrite sym. rewrite <- axiom_12. doesn't work *)
  unfold _mk_sum. assert (h: exists p, r = _dest_sum p).
  exists (inl a). simpl. reflexivity.
  generalize (ε_spec h). set (o := ε (fun p : sum' A B => _dest_sum p = r)).
  auto.
Qed.

Lemma INR_def {A B : Type'} : (@inr A B) = (fun a : B => @_mk_sum A B ((fun a' : B => @CONSTR (prod A B) (N.succ (NUMERAL N0)) (@pair A B (@ε A (fun v : A => True)) a') (fun n : N => @BOTTOM (prod A B))) a)).
Proof.
  apply fun_ext. intro b. apply _dest_sum_inj. simpl.
  match goal with [|- ?x = _] => set (r := x) end.
  (* rewrite sym. rewrite <- axiom_12. doesn't work *)
  unfold _mk_sum. assert (h: exists p, r = _dest_sum p).
  exists (inr(b)). simpl. reflexivity.
  generalize (ε_spec h). set (o := ε (fun p : sum' A B => _dest_sum p = r)).
  auto.
Qed.

(****************************************************************************)
(* Alignment of the option type constructor. *)
(****************************************************************************)

Definition option' (A : Type') := {| type := option A; el := None |}.
Canonical option'.

Definition _dest_option : forall {A : Type'}, option A -> recspace A :=
  fun A o =>
    match o with
    | None => CONSTR (NUMERAL N0) (ε (fun _ => True)) (fun _ => BOTTOM)
    | Some a => CONSTR (N.succ (NUMERAL N0)) a (fun _ => BOTTOM)
    end.

Lemma _dest_option_inj {A : Type'} (o1 o2 : option A) :
  _dest_option o1 = _dest_option o2 -> o1 = o2.
Proof.
  induction o1; induction o2; simpl; rewrite (@CONSTR_INJ A); intros [e1 [e2 e3]].
  rewrite e2. reflexivity. discriminate. discriminate. reflexivity.
Qed.

Definition _mk_option_pred {A : Type'} (r : recspace A) : option A -> Prop :=
  fun o => _dest_option o = r.

Definition _mk_option : forall {A : Type'}, (recspace A) -> option A :=
  fun A r => ε (_mk_option_pred r).

Lemma axiom_13 : forall {A : Type'} (a : option A), (@_mk_option A (@_dest_option A a)) = a.
Proof.
  intros A o. unfold _mk_option.
  match goal with [|- ε ?x = _] => set (Q := x); set (q := ε Q) end.
  assert (i : exists q, Q q). exists o. reflexivity.
  generalize (ε_spec i). fold q. unfold Q, _mk_option_pred. apply _dest_option_inj.
Qed.

Definition option_pred {A : Type'} (r : recspace A) :=
  forall option' : recspace A -> Prop,
      (forall a' : recspace A,
       a' = CONSTR (NUMERAL N0) (ε (fun _ : A => True)) (fun _ : N => BOTTOM) \/
       (exists a'' : A, a' = CONSTR (N.succ (NUMERAL N0)) a'' (fun _ : N => BOTTOM)) ->
       option' a') -> option' r.

Inductive option_ind {A : Type'} : recspace A -> Prop :=
| option_ind0 : option_ind (CONSTR (NUMERAL N0) (ε (fun _ : A => True)) (fun _ : N => BOTTOM))
| option_ind1 a'' : option_ind (CONSTR (N.succ (NUMERAL N0)) a'' (fun _ : N => BOTTOM)).

Lemma option_eq {A : Type'} : @option_pred A = @option_ind A.
Proof.
  apply fun_ext; intro r. apply prop_ext.
  intro h. apply h. intros r' [i|[a'' i]]; subst. apply option_ind0. apply option_ind1.
  induction 1; unfold option_pred; intros r h; apply h.
  left. reflexivity. right. exists a''. reflexivity.
Qed.

Lemma axiom_14' : forall {A : Type'} (r : recspace A), (option_pred r) = ((@_dest_option A (@_mk_option A r)) = r).
Proof.
  intros A r. apply prop_ext.

  intro h. apply (@ε_spec _ (_mk_option_pred r)).
  rewrite option_eq in h. induction h.
  exists None. reflexivity. exists (Some a''). reflexivity.

  intro e. rewrite <- e. intros P h. apply h. destruct (_mk_option r); simpl.
  right. exists t. reflexivity. left. reflexivity.
Qed.

Lemma axiom_14 : forall {A : Type'} (r : recspace A), ((fun a : recspace A => forall option' : (recspace A) -> Prop, (forall a' : recspace A, ((a' = (@CONSTR A (NUMERAL N0) (@ε A (fun v : A => True)) (fun n : N => @BOTTOM A))) \/ (exists a'' : A, a' = ((fun a''' : A => @CONSTR A (N.succ (NUMERAL N0)) a''' (fun n : N => @BOTTOM A)) a''))) -> option' a') -> option' a) r) = ((@_dest_option A (@_mk_option A r)) = r).
Proof. intros A r. apply axiom_14'. Qed.

Lemma NONE_def {A : Type'} : (@None A) = (@_mk_option A (@CONSTR A (NUMERAL N0) (@ε A (fun v : A => True)) (fun n : N => @BOTTOM A))).
Proof.
  apply _dest_option_inj. simpl.
  match goal with [|- ?x = _] => set (r := x) end.
  (* rewrite <- axiom_14'. doesn't work *)
  unfold _mk_option.
  assert (h: exists o, @_mk_option_pred A r o). exists None. reflexivity.
  generalize (ε_spec h).
  set (o := ε (_mk_option_pred r)). unfold _mk_option_pred. auto.
Qed.

Lemma SOME_def {A : Type'} : (@Some A) = (fun a : A => @_mk_option A ((fun a' : A => @CONSTR A (N.succ (NUMERAL N0)) a' (fun n : N => @BOTTOM A)) a)).
Proof.
  apply fun_ext; intro a. apply _dest_option_inj. simpl.
  match goal with [|- ?x = _] => set (r := x) end.
  (* rewrite <- axiom_14'. doesn't work *)
  unfold _mk_option.
  assert (h: exists o, @_mk_option_pred A r o). exists (Some a). reflexivity.
  generalize (ε_spec h).
  set (o := ε (_mk_option_pred r)). unfold _mk_option_pred. auto.
Qed.

(****************************************************************************)
(* Alignment of the list type constructor. *)
(****************************************************************************)

Definition list' (A : Type') := {| type := list A; el := nil |}.
Canonical list'.

Definition FCONS {A : Type'} (a : A) (f: N -> A) (n : N) : A :=
  N.recursion a (fun n _ => f n) n.

Lemma FCONS_def {A : Type'} : @FCONS A = @ε ((prod N (prod N (prod N (prod N N)))) -> A -> (N -> A) -> N -> A) (fun FCONS' : (prod N (prod N (prod N (prod N N)))) -> A -> (N -> A) -> N -> A => forall _17460 : prod N (prod N (prod N (prod N N))), (forall a : A, forall f : N -> A, (FCONS' _17460 a f (NUMERAL N0)) = a) /\ (forall a : A, forall f : N -> A, forall n : N, (FCONS' _17460 a f (N.succ n)) = (f n))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))))).
Proof.
  generalize (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
    (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
      (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
        (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
          NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))); intro p.
  apply fun_ext. intro a. apply fun_ext. intro f. apply fun_ext. intro n.
  match goal with [|- _ = ε ?x _ _ _ _] => set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => @FCONS A).
  unfold Q, FCONS, NUMERAL. intros _. split; intros b g.
  apply N.recursion_0.
  intro x. rewrite N.recursion_succ. reflexivity. reflexivity.
  intros n1 n2 n12 a1 a2 a12. subst n2. subst a2. reflexivity.
  generalize (ε_spec i). intro H. unfold Q at 1 in H.
  generalize (H p); intros [h1 h2].
  destruct (N0_or_succ n) as [|[m j]]; subst n; unfold FCONS.
  rewrite h1. apply N.recursion_0.
  rewrite h2. rewrite N.recursion_succ. reflexivity. reflexivity.
  intros n1 n2 n12 a1 a2 a12. subst n2. subst a2. reflexivity.
Qed.

Fixpoint _dest_list {A : Type'} l : recspace A :=
  match l with
  | nil => CONSTR (NUMERAL N0) (ε (fun _ => True)) (fun _ => BOTTOM)
  | cons a l => CONSTR (N.succ (NUMERAL N0)) a (FCONS (_dest_list l) (fun _ => BOTTOM))
  end.

Definition _mk_list_pred {A : Type'} (r : recspace A) : list A -> Prop :=
  fun l => _dest_list l = r.

Definition _mk_list : forall {A : Type'}, (recspace A) -> list A :=
  fun A r => ε (_mk_list_pred r).

Lemma FCONS_0 {A : Type'} (a : A) (f : N -> A) : FCONS a f (NUMERAL N0) = a.
Proof. reflexivity. Qed.

Lemma _dest_list_inj :
  forall {A : Type'} (l l' : list A), _dest_list l = _dest_list l' -> l = l'.
Proof.
  induction l; induction l'; simpl; rewrite (@CONSTR_INJ A); intros [e1 [e2 e3]].
  reflexivity. discriminate. discriminate. rewrite e2. rewrite (@IHl l'). reflexivity.
  rewrite <- (FCONS_0 (_dest_list l) ((fun _ : N => BOTTOM))).
  rewrite <- (FCONS_0 (_dest_list l') ((fun _ : N => BOTTOM))).
  rewrite e3. reflexivity.
Qed.

Lemma axiom_15 : forall {A : Type'} (a : list A), (@_mk_list A (@_dest_list A a)) = a.
Proof.
  intros A l. unfold _mk_list.
  match goal with [|- ε ?x = _] => set (L' := x); set (l' := ε L') end.
  assert (i : exists l', L' l'). exists l. reflexivity.
  generalize (ε_spec i). fold l'. unfold L', _mk_list_pred. apply _dest_list_inj.
Qed.

Definition list_pred {A : Type'} (r : recspace A) :=
  forall list'0 : recspace A -> Prop,
  (forall a' : recspace A,
  a' = CONSTR (NUMERAL N0) (ε (fun _ : A => True)) (fun _ : N => BOTTOM) \/
  (exists (a0 : A) (a1 : recspace A), a' = CONSTR (N.succ (NUMERAL N0)) a0 (FCONS a1 (fun _ : N => BOTTOM)) /\ list'0 a1) -> list'0 a')
  -> list'0 r.

Inductive list_ind {A : Type'} : recspace A -> Prop :=
| list_ind0 : list_ind (CONSTR (NUMERAL N0) (ε (fun _ : A => True)) (fun _ : N => BOTTOM))
| list_ind1 a'' l'': list_ind (CONSTR (N.succ (NUMERAL N0)) a'' (FCONS (_dest_list l'') (fun _ : N => BOTTOM))).

Lemma list_eq {A : Type'} : @list_pred A = @list_ind A.
Proof.
  apply fun_ext. intro r. apply prop_ext.
  intro h. apply h. intros r' H. destruct H. rewrite H. exact list_ind0. destruct H. destruct H. destruct H. rewrite H. destruct H0.
  assert (_dest_list nil = @CONSTR A (NUMERAL N0) (@ε A (fun v : A => True)) (fun n : N => @BOTTOM A)).
  reflexivity. rewrite <- H0. exact (list_ind1 x nil).
  assert (_dest_list (cons a'' l'') = @CONSTR A (N.succ (NUMERAL N0)) a'' (@FCONS (recspace A) (@_dest_list A l'') (fun n : N => @BOTTOM A))).
  reflexivity. rewrite <- H0. exact (list_ind1 x (a'':: l'')).

  induction 1; unfold list_pred; intros R h; apply h.
  left; reflexivity.
  right. exists a''. exists (_dest_list l''). split. reflexivity. apply h.
  induction l''. auto. right. exists a. exists (_dest_list l''). split. reflexivity.
  apply h. exact IHl''.
Qed.

Lemma axiom_16' : forall {A : Type'} (r : recspace A), (list_pred r) = ((@_dest_list A (@_mk_list A r)) = r).
Proof.
  intros A r. apply prop_ext.

  intro h. apply (@ε_spec _ (_mk_list_pred r)).
  rewrite list_eq in h. induction h.
  exists nil. reflexivity. exists (cons a'' l''). reflexivity.

  intro e. rewrite <- e. intros P h. apply h. destruct (_mk_list r).
  left. reflexivity. right. exists t. exists (_dest_list l). split.
  reflexivity. apply h. generalize l.
  induction l0. left; reflexivity. right. exists a. exists (_dest_list l0). split.
  reflexivity. apply h. exact IHl0.
Qed.

Lemma axiom_16 : forall {A : Type'} (r : recspace A), ((fun a : recspace A => forall list' : (recspace A) -> Prop, (forall a' : recspace A, ((a' = (@CONSTR A (NUMERAL N0) (@ε A (fun v : A => True)) (fun n : N => @BOTTOM A))) \/ (exists a0 : A, exists a1 : recspace A, (a' = ((fun a0' : A => fun a1' : recspace A => @CONSTR A (N.succ (NUMERAL N0)) a0' (@FCONS (recspace A) a1' (fun n : N => @BOTTOM A))) a0 a1)) /\ (list' a1))) -> list' a') -> list' a) r) = ((@_dest_list A (@_mk_list A r)) = r).
Proof. intros A r. apply axiom_16'. Qed.

Lemma NIL_def {A : Type'} : (@nil A) = (@_mk_list A (@CONSTR A (NUMERAL N0) (@ε A (fun v : A => True)) (fun n : N => @BOTTOM A))).
Proof.
  apply _dest_list_inj. simpl.
  match goal with [|- ?x = _] => set (r := x) end.
  (* the sequence rewrite sym. rewrite <- axiom_16' doesn't work *)
  unfold _mk_list.
  assert (h: exists l, @_mk_list_pred A r l). exists nil. reflexivity.
  generalize (ε_spec h).
  set (l := ε (_mk_list_pred r)). unfold _mk_list_pred. auto.
Qed.

Lemma CONS_def {A : Type'} : (@cons A) = (fun a0 : A => fun a1 : list A => @_mk_list A ((fun a0' : A => fun a1' : recspace A => @CONSTR A (N.succ (NUMERAL N0)) a0' (@FCONS (recspace A) a1' (fun n : N => @BOTTOM A))) a0 (@_dest_list A a1))).
Proof.
  apply fun_ext. intro a. apply fun_ext; intro l. apply _dest_list_inj. simpl.
  match goal with [|- ?x = _] => set (r := x) end.
  unfold _mk_list.
  assert (h: exists l', @_mk_list_pred A r l'). exists (cons a l). reflexivity.
  generalize (ε_spec h).
  set (l' := ε (_mk_list_pred r)). unfold _mk_list_pred. auto.
Qed.

Require Import ExtensionalityFacts.

Lemma ISO_def {A B : Type'} : (@is_inverse A B) = (fun _17569 : A -> B => fun _17570 : B -> A => (forall x : B, (_17569 (_17570 x)) = x) /\ (forall y : A, (_17570 (_17569 y)) = y)).
Proof.
  apply fun_ext; intro f. apply fun_ext; intro g.
  unfold is_inverse. apply prop_ext; tauto.
Qed.

Require Import List.

Lemma APPEND_def {A : Type'} : (@app A) = (@ε ((prod N (prod N (prod N (prod N (prod N N))))) -> (list' A) -> (list' A) -> list' A) (fun APPEND' : (prod N (prod N (prod N (prod N (prod N N))))) -> (list A) -> (list A) -> list A => forall _17935 : prod N (prod N (prod N (prod N (prod N N)))), (forall l : list A, (APPEND' _17935 (@nil A) l) = l) /\ (forall h : A, forall t : list A, forall l : list A, (APPEND' _17935 (@cons A h t) l) = (@cons A h (APPEND' _17935 t l)))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))))))))).
Proof.
  apply fun_ext. intro l. simpl.
  match goal with |- _ = ε ?x _ _ => set (Q := x) end.
  assert (i: exists q, Q q). exists (fun _ => @app A). unfold Q. intros. auto.
  generalize (ε_spec i). intro H. symmetry. apply fun_ext. intro l'.
  generalize (NUMERAL (BIT1 32), (NUMERAL 80, (NUMERAL 80, (NUMERAL (BIT1 34), (NUMERAL 78, NUMERAL 68))))); intro p.
  induction l as [|a l]. simpl. apply H.
  assert (ε Q p (a :: l) l' = (a :: (ε Q p l l'))). apply H. simpl. rewrite <- IHl. apply H0.
Qed.

Lemma REVERSE_def {A : Type'} : (@rev A) = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> (list' A) -> list' A) (fun REVERSE' : (prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> (list A) -> list A => forall _17939 : prod N (prod N (prod N (prod N (prod N (prod N N))))), ((REVERSE' _17939 (@nil A)) = (@nil A)) /\ (forall l : list A, forall x : A, (REVERSE' _17939 (@cons A x l)) = (@app A (REVERSE' _17939 l) (@cons A x (@nil A))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))))))))).
Proof.
  apply fun_ext. intro l. simpl.
  match goal with |- _ = ε ?x _ _ => set (Q := x) end.
  assert (i: exists q, Q q). exists (fun _ => @rev A). unfold Q. intros. auto.
  generalize (ε_spec i). intro H. symmetry.
  induction l as [|a l]. simpl. apply H.
  simpl. rewrite <- IHl.
  generalize (NUMERAL 82,
              (NUMERAL (BIT1 34),
                (NUMERAL 86,
                  (NUMERAL (BIT1 34),
                    (NUMERAL 82, (NUMERAL (BIT1 (BIT1 20)),
                      NUMERAL (BIT1 34))))))); intro p.
  assert (ε Q p (a :: l) = (ε Q p l) ++ (a :: nil)). apply H. apply H0.
Qed.

(*Lemma LENGTH_def {A : Type'} : (@length A) = (@ε ((prod N (prod N (prod N (prod N (prod N N))))) -> (list A) -> N) (fun LENGTH' : (prod N (prod N (prod N (prod N (prod N N))))) -> (list A) -> N => forall _17943 : prod N (prod N (prod N (prod N (prod N N)))), ((LENGTH' _17943 (@nil A)) = (NUMERAL N0)) /\ (forall h : A, forall t : list A, (LENGTH' _17943 (@cons A h t)) = (N.succ (LENGTH' _17943 t)))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))))))))).
Proof.
  generalize (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))), (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))), (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))), (NUMERAL (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))), (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))), NUMERAL (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))))))); intro p.
  apply fun_ext. intro l. simpl.
  match goal with |- _ = ε ?x _ _ => set (Q := x) end.
  assert (i: exists q, Q q). exists (fun _ => @length A). unfold Q. auto.
  generalize (ε_spec i). intro H. symmetry.
  induction l. simpl. apply H.
  simpl. rewrite <- IHl. apply H.
Qed.*)

Lemma MAP_def {A B : Type'} : (@map A B) = (@ε ((prod N (prod N N)) -> (A -> B) -> (list' A) -> list' B) (fun MAP' : (prod N (prod N N)) -> (A -> B) -> (list A) -> list B => forall _17950 : prod N (prod N N), (forall f : A -> B, (MAP' _17950 f (@nil A)) = (@nil B)) /\ (forall f : A -> B, forall h : A, forall t : list A, (MAP' _17950 f (@cons A h t)) = (@cons B (f h) (MAP' _17950 f t)))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))).
Proof.
  generalize (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
              (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
                NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))); intro p.
  apply fun_ext. intro f. apply fun_ext. intro l.
  match goal with |- _ = ε ?x _ _ _ => set (Q := x) end.
  assert (i: exists q, Q q). exists (fun _ => @map A B). unfold Q. auto.
  generalize (ε_spec i). intro H. symmetry.
  induction l. simpl. apply H.
  simpl. rewrite <- IHl. apply H.
Qed.

Lemma COND_list {A : Type'} (l0 l1 l2 : list A) :
  match l0 with
  | nil => l1
  | cons h t => l2
  end
  = COND (l0 = nil) l1 l2.
Proof.
  induction l0 as [|a l0]. symmetry. assert ((@nil A = nil) = True). apply prop_ext. auto. auto.
  rewrite H. apply COND_True.
  assert ((a :: l0 = nil) = False). apply prop_ext. intro.
  assert (nil <> a :: l0). apply nil_cons. easy. easy.
  rewrite H. symmetry. apply COND_False.
Qed.

Lemma BUTLAST_def {_25251 : Type'} : (@removelast _25251) = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> (list' _25251) -> list' _25251) (fun BUTLAST' : (prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> (list _25251) -> list _25251 => forall _17958 : prod N (prod N (prod N (prod N (prod N (prod N N))))), ((BUTLAST' _17958 (@nil _25251)) = (@nil _25251)) /\ (forall h : _25251, forall t : list _25251, (BUTLAST' _17958 (@cons _25251 h t)) = (@COND (list' _25251) (t = (@nil _25251)) (@nil _25251) (@cons _25251 h (BUTLAST' _17958 t))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))))))).
Proof.
  generalize (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
              (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
                (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
                  (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
                    (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
                      (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
                        NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))))); intro p.
  apply fun_ext. intro l.
  match goal with |- _ = ε ?x _ _ => set (Q := x) end.
  assert (i: exists q, Q q). exists (fun _ => @removelast _25251). unfold Q. intro. split.
  simpl. reflexivity.
  intros. simpl. apply COND_list.
  generalize (ε_spec i). intro H. symmetry.
  induction l as [|a l]. simpl. apply H.
  assert (ε Q p (a :: l) = COND (l = nil) nil (a :: ε Q p l)).
  apply H. simpl. rewrite <- IHl. transitivity (COND (l = nil) nil (a :: ε Q p l)).
  exact H0. symmetry. apply COND_list.
Qed.

Lemma ALL_def {_25307 : Type'} : (@Forall _25307) = (@ε ((prod N (prod N N)) -> (_25307 -> Prop) -> (list _25307) -> Prop) (fun ALL' : (prod N (prod N N)) -> (_25307 -> Prop) -> (list _25307) -> Prop => forall _17973 : prod N (prod N N), (forall P : _25307 -> Prop, (ALL' _17973 P (@nil _25307)) = True) /\ (forall h : _25307, forall P : _25307 -> Prop, forall t : list _25307, (ALL' _17973 P (@cons _25307 h t)) = ((P h) /\ (ALL' _17973 P t)))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))))))).
Proof.
  generalize (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
    (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
      NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))))); intro p.
  apply fun_ext. intro P. apply fun_ext. intro l.
  match goal with |- _ = ε ?x _ _ _=> set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => @Forall _25307).
  unfold Q. intro. split. intro. apply prop_ext. trivial. intro. apply Forall_nil.
  intros h P0 t. apply prop_ext; apply Forall_cons_iff.
  generalize (ε_spec i). intro. induction l as [|a l]; destruct (H p) as [H1 H2].
  rewrite H1. apply prop_ext. trivial. intro; apply Forall_nil. rewrite H2.
  transitivity (P a /\ Forall P l). apply prop_ext; apply Forall_cons_iff. rewrite IHl. reflexivity.
Qed.

Lemma ForallOrdPairs_nil {A : Type'} (R : A -> A -> Prop) : @ForallOrdPairs A R nil = True.
Proof.
  apply prop_ext. trivial. intro; exact (FOP_nil R).
Qed.

Lemma ForallOrdPairs_hd_tl {A : Type'} (R : A -> A -> Prop) (l : list A) :
  @ForallOrdPairs A R l = ((@Forall A (R (hd (el A) l)) (tl l)) /\ @ForallOrdPairs A R (tl l)).
Proof.
  apply prop_ext. intro. destruct H; simpl. rewrite ForallOrdPairs_nil.
  split. apply Forall_nil. trivial.
  split. exact H. exact H0.
  intro. destruct H as [H1 H2]. destruct l; simpl. rewrite ForallOrdPairs_nil. trivial.
  apply FOP_cons. exact H1. exact H2.
Qed.

Lemma ForallOrdPairs_cons {A : Type'} (R : A -> A -> Prop) (h : A) (t : list A) :
  @ForallOrdPairs A R (h :: t) = ((@Forall A (R h) t) /\ @ForallOrdPairs A R t).
Proof. apply ForallOrdPairs_hd_tl. Qed.

Lemma PAIRWISE_def {A : Type'} : (@ForallOrdPairs A) = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> (A -> A -> Prop) -> (list A) -> Prop) (fun PAIRWISE' : (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> (A -> A -> Prop) -> (list A) -> Prop => forall _18057 : prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))), (forall r : A -> A -> Prop, (PAIRWISE' _18057 r (@nil A)) = True) /\ (forall h : A, forall r : A -> A -> Prop, forall t : list A, (PAIRWISE' _18057 r (@cons A h t)) = ((@Forall A (r h) t) /\ (PAIRWISE' _18057 r t)))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N N)))))) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))))))))))).
Proof.
  generalize (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
    (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
      (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
        (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
          (NUMERAL (BIT1 (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
            (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
              (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
                NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))))))))); intro p.
  apply fun_ext; intro R. apply fun_ext; intro l.
  match goal with |- _ = ε ?x _ _ _=> set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _ => @ForallOrdPairs A).
  unfold Q. intro. split. apply ForallOrdPairs_nil. intros h r t; apply ForallOrdPairs_cons.
  generalize (ε_spec i). intro H. symmetry. induction l as [|a l]. rewrite ForallOrdPairs_nil.
  apply H. rewrite (ForallOrdPairs_cons R a l). rewrite <- IHl. apply H.
Qed.

(* Coercion from bool to Prop, used in the mapping of char to ascii below. *)

Coercion is_true : bool >-> Sortclass.

Lemma is_true_of_true : True = is_true true.
Proof.
  unfold is_true. apply prop_ext. trivial. trivial.
Qed.

Lemma is_true_of_false : False = is_true false.
Proof.
  unfold is_true. apply prop_ext. auto. intro. discriminate.
Qed.

(* Coercion from Prop to bool. *)
(*
Definition bool_of_Prop (P:Prop) : bool := COND P true false.

Coercion bool_of_Prop: Sortclass >-> bool.
*)
(* There are problems for mapping FILTER to List.filter because
HOL-Light's FILTER takes propositional functions as argument while
Coq's List.filter function takes boolean functions as argument. The
error does not occur here but later in the HOL-Light proofs.

Fixpoint filter_bis {A : Type'} (f : A -> Prop) (l : list A) : list A
      := match l with | nil => nil | x :: l => @COND (list A) (f x)
      (x::filter_bis f l) (filter_bis f l) end.

Lemma FILTER_def {_25594 : Type'} : (@filter _25594) = (@ε ((prod N
(prod N (prod N (prod N (prod N N))))) -> (_25594 -> Prop)
-> (list _25594) -> list _25594) (fun FILTER' : (prod N (prod N
(prod N (prod N (prod N N))))) -> (_25594 -> Prop) -> (list
_25594) -> list _25594 => forall _18022 : prod N (prod N (prod N
(prod N (prod N N)))), (forall P : _25594 -> Prop, (FILTER'
_18022 P (@nil _25594)) = (@nil _25594)) /\ (forall h : _25594, forall
P : _25594 -> Prop, forall t : list _25594, (FILTER' _18022 P (@cons
_25594 h t)) = (@COND (list _25594) (P h) (@cons _25594 h (FILTER'
_18022 P t)) (FILTER' _18022 P t)))) (@pair N (prod N (prod N
(prod N (prod N N)))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0
(BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N)))
(NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair
N (prod N (prod N N)) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0
(BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT0 (BIT0
(BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1
(BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT1
(BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))))))).  Proof.  generalize
(NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))), (NUMERAL
(BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0))))))), (NUMERAL (BIT0
(BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))), (NUMERAL (BIT0 (BIT0
(BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))), (NUMERAL (BIT1 (BIT0 (BIT1
(BIT0 (BIT0 (BIT0 (BIT1 0))))))), NUMERAL (BIT0 (BIT1 (BIT0 (BIT0
(BIT1 (BIT0 (BIT1 0)))))))))))); intro p.  apply fun_ext; intro
f. apply fun_ext; intro l.  match goal with |- _ = ε ?x _ _ _=> set (Q
:= x) end.  assert (i : exists q, Q q). exists (fun _=> @filter_bis
_25594).  unfold Q. intro. auto.  generalize (ε_spec i). intro
H. symmetry. induction l; simpl. apply H.  assert (ε Q p (fun x :
_25594 => f x) (a :: l) = COND (f a) (a::ε Q p (fun x : _25594 => f x)
l) (ε Q p (fun x : _25594 => f x) l )).  apply H. transitivity (COND
(f a) (a :: ε Q p (fun x : _25594 => f x) l) (ε Q p (fun x : _25594 =>
f x) l)).  exact H0. transitivity (COND (f a) (a :: ε Q p (fun x :
_25594 => f x) l) (filter f l)).  rewrite <- IHl. reflexivity.
destruct (f a). rewrite <- is_true_of_true. rewrite COND_True. rewrite
<- IHl. reflexivity.  rewrite <- is_true_of_false. apply COND_False.
Qed.*)

Lemma MEM_def {_25376 : Type'} : (@In _25376) = (@ε ((prod N (prod N N)) -> _25376 -> (list _25376) -> Prop) (fun MEM' : (prod N (prod N N)) -> _25376 -> (list _25376) -> Prop => forall _17995 : prod N (prod N N), (forall x : _25376, (MEM' _17995 x (@nil _25376)) = False) /\ (forall h : _25376, forall x : _25376, forall t : list _25376, (MEM' _17995 x (@cons _25376 h t)) = ((x = h) \/ (MEM' _17995 x t)))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))))))).
Proof.
  generalize (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
    (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
      NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))))); intro p.
  apply fun_ext; intro x. apply fun_ext; intro l.
  match goal with |- _ = ε ?x _ _ _=> set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _=> @In _25376). unfold Q. intro. simpl.
  split. trivial. intros. apply prop_ext. intro. destruct H. symmetry in H. left. exact H. right. exact H.
  intro. destruct H. left. symmetry in H. exact H. right. exact H.
  generalize (ε_spec i). intro H. symmetry. induction l as [|a l]; simpl. apply H. rewrite <- IHl.
  transitivity ((x = a \/ ε Q p x l)). apply H. apply prop_ext.
  intro. destruct H0. left. symmetry. exact H0. right. exact H0.
  intro. destruct H0. left. symmetry. exact H0. right. exact H0.
Qed.

(*Definition repeat_with_perm_args {A: Type'} (n: N) (a: A) := @repeat A a n.

Lemma REPLICATE_def {_25272 : Type'} : (@repeat_with_perm_args _25272) = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))))) -> N -> _25272 -> list _25272) (fun REPLICATE' : (prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))))) -> N -> _25272 -> list _25272 => forall _17962 : prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))), (forall x : _25272, (REPLICATE' _17962 (NUMERAL N0) x) = (@nil _25272)) /\ (forall n : N, forall x : _25272, (REPLICATE' _17962 (N.succ n) x) = (@cons _25272 x (REPLICATE' _17962 n x)))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N N)))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))))))))))).
Proof.
  generalize (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
    (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
      (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
        (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
          (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
            (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
              (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0))))))),
                (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
                  NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))))))))); intro p.
  apply fun_ext; intro n. apply fun_ext; intro a.
  match goal with |- _ = ε ?x _ _ _=> set (Q := x) end.
  assert (i : exists q, Q q). exists (fun _=> @repeat_with_perm_args _25272).
  unfold Q. intro; simpl. auto.
  generalize (ε_spec i). intro H. symmetry. induction n; simpl. apply H.
  rewrite <- IHn. apply H.
Qed.*)

(*
Definition fold_right_with_perm_args {A B : Type'} (f: A -> B -> B) (l: list A) (b: B) : B := @fold_right B A f b l.

Lemma ITLIST_def {A B : Type'} : (@fold_right_with_perm_args A B) = (@ε ((prod N (prod N (prod N (prod N (prod N N))))) -> (A -> B -> B) -> (list A) -> B -> B) (fun ITLIST' : (prod N (prod N (prod N (prod N (prod N N))))) -> (A -> B -> B) -> (list A) -> B -> B => forall _18151 : prod N (prod N (prod N (prod N (prod N N)))), (forall f : A -> B -> B, forall b : B, (ITLIST' _18151 f (@nil A) b) = b) /\ (forall h : A, forall f : A -> B -> B, forall t : list A, forall b : B, (ITLIST' _18151 f (@cons A h t) b) = (f h (ITLIST' _18151 f t b)))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))))))).
Proof.
  generalize (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
    (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
      (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
        (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
          (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
            NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))))); intro p.
  apply fun_ext; intro f. apply fun_ext; intro l. apply fun_ext; intro a.
  match goal with |- _ = ε ?x _ _ _ _ => set (Q := x) end.
  assert (i: exists q, Q q). exists (fun _ => @fold_right_with_perm_args A B).
  unfold Q. intro. simpl. auto.
  generalize (ε_spec i). intro H. symmetry. induction l; simpl. apply H.
  rewrite <- IHl. apply H.
Qed.
*)

Definition HD {A : Type'} := @ε ((prod N N) -> (list A) -> A) (fun HD' : (prod N N) -> (list A) -> A => forall _17927 : prod N N, forall t : list A, forall h : A, (HD' _17927 (@cons A h t)) = h) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))).

Lemma HD_of_cons {A: Type'} (h: A) (t: list A) : @HD A (h :: t) = h.
Proof.
  unfold HD. generalize (NUMERAL (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
    NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))); intro p.
  match goal with |- ε ?x _ _ = _=> set (Q := x) end.
  assert (i: exists q, Q q). exists (fun _=> @hd A (HD nil)).
  unfold Q. intro. simpl. trivial.
  generalize (ε_spec i). intro H. apply H.
Qed.

Definition hd {A:Type'} := @hd A (HD nil).

Lemma HD_def {A : Type'} : @hd A = @HD A.
Proof.
  apply fun_ext. intro l. unfold hd, HD.
  generalize (NUMERAL (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0))))))),
    NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))); intro p.
  match goal with |- _ = ε ?x _ _=> set (Q := x) end.
  assert (i: exists q, Q q). exists (fun _ => @hd A).
  unfold Q. intro. simpl. trivial.
  generalize (ε_spec i). intro H. destruct l; simpl. reflexivity. rewrite H. reflexivity.
Qed.

Definition TL {A : Type'} := (@ε ((prod N N) -> (list A) -> list A) (fun TL' : (prod N N) -> (list A) -> list A => forall _17931 : prod N N, forall h : A, forall t : list A, (TL' _17931 (@cons A h t)) = t) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))))).

Definition tl {A : Type'} (l : list A) :=
match l with
| nil => @TL A nil
| cons h t => @tl A (cons h t)
end.

Lemma TL_def {A : Type'} : @tl A = @TL A.
Proof.
  apply fun_ext. intro l. destruct l. simpl. reflexivity. unfold TL.
  generalize (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))),
    NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))); intro p.
  match goal with |-_ = ε ?x _ _ => set (Q := x) end.
  assert (i: exists q, Q q). exists (fun _=> @tl A).
  unfold Q. intro. simpl. trivial.
  generalize (ε_spec i). intro H.
  unfold Q. simpl. symmetry. apply H.
Qed.

(* We cannot map EL to List.nth because the equation defining EL
requires (TL NIL) to be equal to NIL, which is not the case.

Lemma nth_of_0 {A: Type'} (l: list A) d : nth (NUMERAL N0) l d =
List.hd d l.  Proof. destruct l;
simpl. reflexivity. symmetry. reflexivity. Qed.

Lemma nth_of_Suc {A: Type'} (n: N) (l: list A) d : nth (N.succ n) l d =
nth n (List.tl l) d.  Proof. destruct l; simpl. destruct n; simpl;
reflexivity. reflexivity. Qed.

Definition EL {A: Type'} (n: N) (l: list A) : A := @nth A n l (HD
nil).

Lemma EL_def {_25569 : Type'} : (@EL _25569) = (@ε ((prod N N) ->
N -> (list _25569) -> _25569) (fun EL' : (prod N N) -> N ->
(list _25569) -> _25569 => forall _18015 : prod N N, (forall l :
list _25569, (EL' _18015 (NUMERAL N0) l) = (@hd _25569 l)) /\ (forall n
: N, forall l : list _25569, (EL' _18015 (N.succ n) l) = (EL' _18015 n
(@tl _25569 l)))) (@pair N N (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0
(BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0
(BIT0 (BIT1 0)))))))))).  Proof.  generalize (NUMERAL (BIT1 (BIT0
(BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))), NUMERAL (BIT0 (BIT0 (BIT1
(BIT1 (BIT0 (BIT0 (BIT1 0)))))))); intro p.  apply fun_ext. intro n.
match goal with |-_ = ε ?x _ _ => set (Q := x) end.  assert (i: exists
q, Q q). exists (fun _ => @EL _25569).  unfold Q. intro. unfold
EL. simpl. split.  destruct l; reflexivity. intros n' l. rewrite
nth_of_Suc.  generalize (ε_spec i). intro H. unfold EL. apply fun_ext.
induction n; simpl; intro l.  rewrite nth_of_0. symmetry. apply H.
rewrite nth_of_Suc. rewrite (IHn (tl l)). symmetry. apply H.  Qed.*)

(****************************************************************************)
(* Alignment of the type of ASCII characters. *)
(****************************************************************************)

(* Note the mismatch between Coq's ascii which takes booleans as arguments
and HOL-Light's char which takes propositions as arguments. *)

Require Import Coq.Strings.Ascii.

Definition ascii' := {| type := ascii; el := zero |}.
Canonical ascii'.

Definition _dest_char : ascii -> recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))) :=
fun a => match a with
| Ascii a0 a1 a2 a3 a4 a5 a6 a7 => (fun a0' : Prop => fun a1' : Prop => fun a2' : Prop => fun a3' : Prop => fun a4' : Prop => fun a5' : Prop => fun a6' : Prop => fun a7' : Prop => @CONSTR (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))) (NUMERAL N0) (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))) a0' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))) a1' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))) a2' (@pair Prop (prod Prop (prod Prop (prod Prop Prop))) a3' (@pair Prop (prod Prop (prod Prop Prop)) a4' (@pair Prop (prod Prop Prop) a5' (@pair Prop Prop a6' a7'))))))) (fun n : N => @BOTTOM (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))))) a0 a1 a2 a3 a4 a5 a6 a7
end.

Definition _mk_char_pred (r : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))) : ascii -> Prop :=
  fun a => _dest_char a = r.

Definition _mk_char : (recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))) -> ascii :=
  fun r => ε (_mk_char_pred r).

Lemma is_true_inj (b b' : bool) : is_true b = is_true b' -> b = b'.
Proof.
  intro. induction b; induction b'.
  reflexivity.
  unfold is_true in H. symmetry. rewrite <- H. reflexivity.
  unfold is_true in H. rewrite H; reflexivity.
  reflexivity.
Qed.

Lemma _dest_char_inj (a a' : ascii) : _dest_char a = _dest_char a' -> a = a'.
Proof.
  induction a. induction a'. simpl.
  rewrite (@CONSTR_INJ (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))).
  intros [e1 [e2 e3]].
  assert (b = b7 /\ b0 = b8 /\ b1 = b9 /\ b2 = b10 /\ b3 = b11 /\ b4 = b12 /\ b5 = b13 /\ b6 = b14).
  apply pair_equal_spec in e2. repeat (rewrite pair_equal_spec in e2; split).
  apply is_true_inj; apply e2.
  apply is_true_inj; apply e2.
  apply is_true_inj; apply e2.
  apply is_true_inj; apply e2.
  apply is_true_inj; apply e2.
  apply is_true_inj; apply e2. split.
  apply is_true_inj; apply e2.
  apply is_true_inj; apply e2.
  destruct H; rewrite H. destruct H0; rewrite H0. destruct H1; rewrite H1. destruct H2; rewrite H2. destruct H3; rewrite H3.
  destruct H4; rewrite H4. destruct H5; rewrite H5. rewrite H6. reflexivity.
Qed.

Lemma axiom_17 : forall (a : ascii), (_mk_char (_dest_char a)) = a.
Proof.
  intro a. unfold _mk_char.
  match goal with [|- ε ?x = _] => set (A' := x); set (a' := ε A') end.
  assert (i : exists a', A' a'). exists a. reflexivity.
  generalize (ε_spec i). fold a'. unfold A', _mk_char_pred. apply _dest_char_inj.
Qed.

Definition char_pred (r : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))) :=
  forall char' : (recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))) -> Prop, (forall a' : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))), (exists a0 : Prop, exists a1 : Prop, exists a2 : Prop, exists a3 : Prop, exists a4 : Prop, exists a5 : Prop, exists a6 : Prop, exists a7 : Prop, a' =
    ((fun a0' : Prop => fun a1' : Prop => fun a2' : Prop => fun a3' : Prop => fun a4' : Prop => fun a5' : Prop => fun a6' : Prop => fun a7' : Prop => @CONSTR (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))) (NUMERAL N0) (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))) a0' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))) a1' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))) a2' (@pair Prop (prod Prop (prod Prop (prod Prop Prop))) a3' (@pair Prop (prod Prop (prod Prop Prop)) a4' (@pair Prop (prod Prop Prop) a5' (@pair Prop Prop a6' a7'))))))) (fun n : N => @BOTTOM (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))))) a0 a1 a2 a3 a4 a5 a6 a7)) -> char' a') -> char' r.

Inductive char_ind : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))) -> Prop :=
| char_ind_cons a0 a1 a2 a3 a4 a5 a6 a7 :
  char_ind (CONSTR (NUMERAL N0) (is_true a0, (is_true a1, (is_true a2, (is_true a3, (is_true a4, (is_true a5, (is_true a6, is_true a7))))))) (fun _ => BOTTOM)).

Lemma Prop_bool_eq (P : Prop) : P = COND P true false.
Proof.
  case (prop_degen P); intro H; rewrite H.
  rewrite COND_True, is_true_of_true. reflexivity.
  rewrite COND_False, is_true_of_false. reflexivity.
Qed.

Lemma char_eq : char_pred = char_ind.
Proof.
  apply fun_ext; intro r. apply prop_ext.
  intro h. apply h. intros r' [a0 [a1 [a2 [a3 [a4 [a5 [a6 [a7 e]]]]]]]].
  rewrite e, (Prop_bool_eq a0), (Prop_bool_eq a1), (Prop_bool_eq a2),
    (Prop_bool_eq a3), (Prop_bool_eq a4), (Prop_bool_eq a5), (Prop_bool_eq a6),
    (Prop_bool_eq a7).
  exact (char_ind_cons (COND a0 true false) (COND a1 true false)
           (COND a2 true false) (COND a3 true false) (COND a4 true false)
           (COND a5 true false) (COND a6 true false) (COND a7 true false)).
  induction 1. unfold char_pred. intros R h. apply h.
  exists a0. exists a1. exists a2. exists a3. exists a4. exists a5. exists a6. exists a7. reflexivity.
Qed.

Lemma axiom_18' : forall (r : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))),
char_pred r = ((_dest_char (_mk_char r)) = r).
Proof.
  intro r. apply prop_ext.

  intro h. apply (@ε_spec _ (_mk_char_pred r)).
  rewrite char_eq in h. induction h. exists (Ascii a0 a1 a2 a3 a4 a5 a6 a7). reflexivity.

  intro e. rewrite <- e. intros P h. apply h. destruct (_mk_char r); simpl.
  exists (is_true b). exists (is_true b0). exists (is_true b1). exists (is_true b2). exists (is_true b3). exists (is_true b4). exists (is_true b5). exists (is_true b6).
  reflexivity.
Qed.

Lemma axiom_18 : forall (r : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))), ((fun a : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))) => forall char' : (recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))) -> Prop, (forall a' : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))), (exists a0 : Prop, exists a1 : Prop, exists a2 : Prop, exists a3 : Prop, exists a4 : Prop, exists a5 : Prop, exists a6 : Prop, exists a7 : Prop, a' =
((fun a0' : Prop => fun a1' : Prop => fun a2' : Prop => fun a3' : Prop => fun a4' : Prop => fun a5' : Prop => fun a6' : Prop => fun a7' : Prop => @CONSTR (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))) (NUMERAL N0) (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))) a0' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))) a1' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))) a2' (@pair Prop (prod Prop (prod Prop (prod Prop Prop))) a3' (@pair Prop (prod Prop (prod Prop Prop)) a4' (@pair Prop (prod Prop Prop) a5' (@pair Prop Prop a6' a7'))))))) (fun n : N => @BOTTOM (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))))) a0 a1 a2 a3 a4 a5 a6 a7)) -> char' a') -> char' a) r) = ((_dest_char (_mk_char r)) = r).
Proof. intro r. apply axiom_18'. Qed.

(*****************************************************************************)
(* Alignment of the type nadd of nearly additive sequences of naturals. *)
(*****************************************************************************)

Definition dist := fun p : prod N N => N.add (N.sub (@fst N N p) (@snd N N p)) (N.sub (@snd N N p) (@fst N N p)).

Lemma DIST_REFL : forall n : N, dist (n,n) = 0.
Proof. intro n. unfold dist. simpl. rewrite N.sub_diag. reflexivity. Qed.

Lemma DIST_SYM x y : dist (x,y) = dist (y,x).
Proof. unfold dist; simpl. lia. Qed.

Lemma DIST_TRIANGLE x y z : dist (x,z) <= dist (x,y) + dist (y,z).
Proof. unfold dist; simpl. lia. Qed.

Definition is_nadd := fun f : N -> N => exists B : N, forall m : N, forall n : N, N.le (dist (@pair N N (N.mul m (f n)) (N.mul n (f m)))) (N.mul B (N.add m n)).

Lemma is_nadd_times n : is_nadd (fun x => n * x).
Proof.
  exists 0. intros x y. simpl. assert (e: x*(n*y)=y*(n*x)). lia.
  rewrite e, DIST_REFL. reflexivity.
Qed.

Lemma is_nadd_0 : is_nadd (fun _ => 0).
Proof. exact (is_nadd_times 0). Qed.

Definition nadd := subtype is_nadd_0.

Definition mk_nadd := mk is_nadd_0.
Definition dest_nadd := dest is_nadd_0.

Lemma axiom_19 : forall (a : nadd), (mk_nadd (dest_nadd a)) = a.
Proof. exact (mk_dest is_nadd_0). Qed.

Lemma axiom_20_aux : forall (r : N -> N), (is_nadd r) -> ((dest_nadd (mk_nadd r)) = r).
Proof. exact (dest_mk_aux is_nadd_0). Qed.

Lemma axiom_20 : forall (r : N -> N), (is_nadd r) = ((dest_nadd (mk_nadd r)) = r).
Proof. exact (dest_mk is_nadd_0). Qed.

Definition nadd_of_num : N -> nadd := fun _23288 : N => mk_nadd (fun n : N => N.mul _23288 n).

Definition nadd_le : nadd -> nadd -> Prop := fun _23295 : nadd => fun _23296 : nadd => exists B : N, forall n : N, N.le (dest_nadd _23295 n) (N.add (dest_nadd _23296 n) B).

Lemma nadd_le_refl x : nadd_le x x.
Proof. exists 0. intro n. lia. Qed.

Lemma  nadd_le_trans x y z : nadd_le x y -> nadd_le y z -> nadd_le x z.
Proof.
  intros [B h] [C i]. exists (B+C). intro n. generalize (h n). generalize (i n). lia.
Qed.

Add Relation _ nadd_le
    reflexivity proved by nadd_le_refl
    transitivity proved by nadd_le_trans
as nadd_le_rel.

Definition nadd_add : nadd -> nadd -> nadd := fun _23311 : nadd => fun _23312 : nadd => mk_nadd (fun n : N => N.add (dest_nadd _23311 n) (dest_nadd _23312 n)).

Lemma is_nadd_add_aux f g : is_nadd f -> is_nadd g -> is_nadd (fun n => f n + g n).
Proof.
  intros [b i] [c j]. exists (b+c). intros x y.
  generalize (i x y); intro ixy. generalize (j x y); intro jxy.
  unfold dist in *; simpl in *. lia.
Qed.

Lemma is_nadd_add f g : is_nadd (fun n => dest_nadd f n + dest_nadd g n).
Proof.
  destruct f as [f hf]. destruct g as [g hg]. simpl.
  apply is_nadd_add_aux. exact hf. exact hg.
Qed.

Lemma nadd_of_num_add p q :
  nadd_of_num (p + q) = nadd_add (nadd_of_num p) (nadd_of_num q).
Proof.
  unfold nadd_add, nadd_of_num. f_equal. apply fun_ext; intro x.
  rewrite axiom_20_aux. 2: apply is_nadd_times.
  rewrite axiom_20_aux. 2: apply is_nadd_times.
  lia.
Qed.

Lemma NADD_ADD_SYM p q : nadd_add p q = nadd_add q p.
Proof. unfold nadd_add. f_equal. apply fun_ext; intro x. lia. Qed.

Lemma NADD_ADD_ASSOC p q r :
  nadd_add (nadd_add p q) r = nadd_add p (nadd_add q r).
Proof.
  unfold nadd_add. f_equal. apply fun_ext; intro x. rewrite !axiom_20_aux. lia.
  apply is_nadd_add. apply is_nadd_add.
Qed.

Definition nadd_mul : nadd -> nadd -> nadd := fun _23325 : nadd => fun _23326 : nadd => mk_nadd (fun n : N => dest_nadd _23325 (dest_nadd _23326 n)).

Definition nadd_rinv : nadd -> N -> N := fun _23462 : nadd => fun n : N => N.div (N.mul n n) (dest_nadd _23462 n).

Definition nadd_eq : nadd -> nadd -> Prop := fun _23276 : nadd => fun _23277 : nadd => exists B : N, forall n : N, N.le (dist (@pair N N (dest_nadd _23276 n) (dest_nadd _23277 n))) B.

Lemma NADD_EQ_REFL f : nadd_eq f f.
Proof. unfold nadd_eq. exists 0. intro n. unfold dist; simpl. lia. Qed.

Lemma nadd_eq_sym f g : nadd_eq f g -> nadd_eq g f.
Proof. intros [b fg]. exists b. intro n. rewrite DIST_SYM. apply fg. Qed.

Lemma nadd_eq_trans f g h : nadd_eq f g -> nadd_eq g h -> nadd_eq f h.
Proof.
  intros [b fg] [c gh]. exists (b+c). intro n.
  rewrite DIST_TRIANGLE with (y := dest_nadd g n).
  generalize (fg n); intro fgn. generalize (gh n); intro ghn.
  transitivity (b + dist (dest_nadd g n, dest_nadd h n)). lia.
  transitivity (b+c); lia.
Qed.

Add Relation _ nadd_eq
    reflexivity proved by NADD_EQ_REFL
    symmetry proved by nadd_eq_sym
    transitivity proved by nadd_eq_trans
as nadd_eq_rel.

Require Import Setoid.

Add Morphism nadd_add
    with signature nadd_eq ==> nadd_eq ==> nadd_eq
      as nadd_add_morph.
Proof.
  intros f f' [b ff'] g g' [c gg']. exists (b+c). intro n.
  generalize (ff' n); intro ff'n. generalize (gg' n); intro gg'n.
  unfold nadd_add. rewrite !axiom_20_aux. unfold dist in *; simpl in *. lia.
  apply is_nadd_add. apply is_nadd_add.
Qed.

(*Add Morphism nadd_le
    with signature nadd_eq ==> nadd_eq ==> iff
      as nadd_le_morph.
Proof.
  intros f f' [b ff'] g g' [c gg'].
Abort.*)

Require Import ProofIrrelevance.

Lemma nadd_add_lcancel x y z : nadd_add x y = nadd_add x z -> y = z.
Proof.
  intro h. destruct x as [x hx]. destruct y as [y hy]. destruct z as [z hz].
  apply subset_eq_compat. unfold nadd_add in h. simpl in h. apply mk_inj in h.
  apply fun_ext; intro a. generalize (ext_fun h a); simpl; intro ha. lia.
  apply is_nadd_add_aux; assumption. apply is_nadd_add_aux; assumption.
Qed.

Lemma NADD_ADD_LCANCEL x y z :
  nadd_eq (nadd_add x y ) (nadd_add x z) -> nadd_eq y z.
Proof.
  intro h. destruct x as [x hx]. destruct y as [y hy]. destruct z as [z hz].
  destruct h as [B h]. exists B. intro n. generalize (h n). unfold nadd_add. simpl.
  unfold dest_nadd, mk_nadd. rewrite !dest_mk_aux. unfold dist. simpl. lia.
  apply is_nadd_add_aux; assumption. apply is_nadd_add_aux; assumption.
Qed.

Definition nadd_inv : nadd -> nadd := fun _23476 : nadd => @COND nadd (nadd_eq _23476 (nadd_of_num (NUMERAL N0))) (nadd_of_num (NUMERAL N0)) (mk_nadd (nadd_rinv _23476)).

(*****************************************************************************)
(* Alignment of the type hreal of non-negative real numbers. *)
(*****************************************************************************)

Definition hreal := quotient nadd_eq.

Definition mk_hreal := mk_quotient nadd_eq.
Definition dest_hreal := dest_quotient nadd_eq.

Lemma axiom_21 : forall (a : hreal), (mk_hreal (dest_hreal a)) = a.
Proof. exact (mk_dest_quotient nadd_eq). Qed.

Lemma axiom_22_aux : forall r : nadd -> Prop, (exists x : nadd, r = nadd_eq x) -> dest_hreal (mk_hreal r) = r.
Proof. exact (dest_mk_aux_quotient nadd_eq). Qed.

Lemma axiom_22 : forall (r : nadd -> Prop), ((fun s : nadd -> Prop => exists x : nadd, s = (nadd_eq x)) r) = ((dest_hreal (mk_hreal r)) = r).
Proof. exact (dest_mk_quotient nadd_eq). Qed.

Definition hreal_of_num : N -> hreal := fun m : N => mk_hreal (nadd_eq (nadd_of_num m)).

Definition hreal_add : hreal -> hreal -> hreal := fun x : hreal => fun y : hreal => mk_hreal (fun u : nadd => exists x' : nadd, exists y' : nadd, (nadd_eq (nadd_add x' y') u) /\ ((dest_hreal x x') /\ (dest_hreal y y'))).

Lemma hreal_add_of_num p q :
  hreal_of_num (p + q) = hreal_add (hreal_of_num p) (hreal_of_num q).
Proof.
  unfold hreal_add, hreal_of_num. f_equal. apply fun_ext; intro x.
  apply prop_ext; intro h.
  exists (nadd_of_num p). exists (nadd_of_num q). split.
  rewrite <- nadd_of_num_add. exact h. split.
  rewrite axiom_22_aux. 2: exists (nadd_of_num p); reflexivity. apply NADD_EQ_REFL.
  rewrite axiom_22_aux. 2: exists (nadd_of_num q); reflexivity. apply NADD_EQ_REFL.
  destruct h as [f [g [h1 [h2 h3]]]].
  rewrite axiom_22_aux in h2. 2: exists (nadd_of_num p); reflexivity.
  rewrite axiom_22_aux in h3. 2: exists (nadd_of_num q); reflexivity.
  rewrite nadd_of_num_add. rewrite h2, h3. exact h1.
Qed.

Lemma succ_eq_add_1 n : N.succ n = n + 1. Proof. lia. Qed.

Lemma hreal_of_num_S n : hreal_of_num (N.succ n) = hreal_add (hreal_of_num n) (hreal_of_num 1).
Proof. rewrite succ_eq_add_1, hreal_add_of_num. reflexivity. Qed.

Lemma hreal_add_sym p q : hreal_add p q = hreal_add q p.
Proof.
  unfold hreal_add. f_equal. apply fun_ext; intro x.
  apply prop_ext; intros [y [z [h1 [h2 h3]]]].
  exists z. exists y. split. rewrite NADD_ADD_SYM. exact h1. auto.
  exists z. exists y. split. rewrite NADD_ADD_SYM. exact h1. auto.
Qed.

Lemma hreal_add_of_mk_hreal p q :
  hreal_add (mk_hreal (nadd_eq p)) (mk_hreal (nadd_eq q))
  = mk_hreal (nadd_eq (nadd_add p q)).
Proof.
  unfold hreal_add. apply f_equal. apply fun_ext; intro x.
  apply prop_ext; intro h.

  unfold dest_hreal, mk_hreal in h. destruct h as [p' [q' [h1 [h2 h3]]]].
  rewrite dest_mk_aux_quotient in h2. 2: apply is_eq_class_of.
  rewrite dest_mk_aux_quotient in h3. 2: apply is_eq_class_of.
  rewrite h2, h3. exact h1.

  exists p. exists q. split. exact h. unfold dest_hreal, mk_hreal.
  rewrite !dest_mk_aux_quotient. split; reflexivity.
  apply is_eq_class_of. apply is_eq_class_of.
Qed.

Lemma mk_hreal_nadd_eq p : mk_hreal (nadd_eq (elt_of p)) = p.
Proof.
  unfold mk_hreal. apply mk_quotient_elt_of.
  apply NADD_EQ_REFL. apply nadd_eq_sym. apply nadd_eq_trans.
Qed.

(*Lemma hreal_add_is_mk_hreal p q :
  hreal_add p q = mk_hreal (nadd_eq (nadd_add (elt_of p) (elt_of q))).
Proof.
  rewrite <- (mk_hreal_nadd_eq p), <- (mk_hreal_nadd_eq q), hreal_add_of_mk_hreal.
  unfold mk_hreal at 3. unfold mk_hreal at 3. rewrite !mk_quotient_elt_of.
  reflexivity.
  apply NADD_EQ_REFL. apply nadd_eq_sym. apply nadd_eq_trans.
  apply NADD_EQ_REFL. apply nadd_eq_sym. apply nadd_eq_trans.
Qed.*)

Lemma hreal_add_assoc p q r :
  hreal_add (hreal_add p q) r = hreal_add p (hreal_add q r).
Proof.
  rewrite <- (mk_hreal_nadd_eq p), <- (mk_hreal_nadd_eq q),
    <- (mk_hreal_nadd_eq r), !hreal_add_of_mk_hreal.
  f_equal. rewrite NADD_ADD_ASSOC. reflexivity.
Qed.

Lemma hreal_add_lcancel p q r : hreal_add p r = hreal_add q r -> p = q.
Proof.
  rewrite <- (mk_hreal_nadd_eq p), <- (mk_hreal_nadd_eq q),
    <- (mk_hreal_nadd_eq r), !hreal_add_of_mk_hreal; intro e.
  unfold mk_hreal, mk_quotient in e. apply mk_inj in e.
  2: apply is_eq_class_of. 2: apply is_eq_class_of.
  apply eq_class_elim in e. 2: apply NADD_EQ_REFL.
  rewrite NADD_ADD_SYM, (NADD_ADD_SYM (elt_of q)) in e.
  apply NADD_ADD_LCANCEL in e.
  f_equal. apply eq_class_intro. apply nadd_eq_sym. apply nadd_eq_trans.
  exact e.
Qed.

Definition hreal_mul : hreal -> hreal -> hreal := fun x : hreal => fun y : hreal => mk_hreal (fun u : nadd => exists x' : nadd, exists y' : nadd, (nadd_eq (nadd_mul x' y') u) /\ ((dest_hreal x x') /\ (dest_hreal y y'))).

Definition hreal_le : hreal -> hreal -> Prop := fun x : hreal => fun y : hreal => @ε Prop (fun u : Prop => exists x' : nadd, exists y' : nadd, ((nadd_le x' y') = u) /\ ((dest_hreal x x') /\ (dest_hreal y y'))).

(*Lemma hreal_le_refl x : hreal_le x x.
Proof.
  unfold hreal_le.
  match goal with [|- ε ?x] => set (Q := x); set (q := ε Q) end.
  assert (i: exists x, Q x). exists True. set (t := elt_of x). exists t. exists t. split.
  rewrite is_True. apply nadd_le_refl.
  assert (h: dest_hreal x t). apply dest_quotient_elt_of. apply NADD_EQ_REFL.
  auto.
  generalize (ε_spec i); intros [x1 [x2 [h1 [h2 h3]]]].
  unfold reverse_coercion. rewrite <- h1.
  apply dest_quotient_elim in h2.
  2: apply NADD_EQ_REFL. 2: apply nadd_eq_sym. 2: apply nadd_eq_trans.
  apply dest_quotient_elim in h3.
  2: apply NADD_EQ_REFL. 2: apply nadd_eq_sym. 2: apply nadd_eq_trans.
  rewrite <- h2, <- h3. reflexivity.
Qed.

Add Relation _ hreal_le
    reflexivity proved by hreal_le_refl
    (*transitivity proved by hreal_le_trans*)
as hreal_le_rel.*)

Definition hreal_inv : hreal -> hreal := fun x : hreal => mk_hreal (fun u : nadd => exists x' : nadd, (nadd_eq (nadd_inv x') u) /\ (dest_hreal x x')).

(*****************************************************************************)
(* Operations on treal (pairs of hreal's). *)
(*****************************************************************************)

Definition treal_of_num : N -> prod hreal hreal := fun _23721 : N => @pair hreal hreal (hreal_of_num _23721) (hreal_of_num (NUMERAL N0)).

Definition treal_neg : (prod hreal hreal) -> prod hreal hreal := fun _23726 : prod hreal hreal => @pair hreal hreal (@snd hreal hreal _23726) (@fst hreal hreal _23726).

Definition treal_add : (prod hreal hreal) -> (prod hreal hreal) -> prod hreal hreal := fun _23735 : prod hreal hreal => fun _23736 : prod hreal hreal => @pair hreal hreal (hreal_add (@fst hreal hreal _23735) (@fst hreal hreal _23736)) (hreal_add (@snd hreal hreal _23735) (@snd hreal hreal _23736)).

Lemma treal_add_of_num p q :
  treal_of_num (p + q) = treal_add (treal_of_num p) (treal_of_num q).
Proof.
  unfold treal_of_num, treal_add; simpl.
  f_equal; rewrite <- hreal_add_of_num; reflexivity.
Qed.

Lemma treal_add_sym  p q : treal_add p q = treal_add q p.
Proof. unfold treal_add. f_equal; apply hreal_add_sym. Qed.

Definition treal_mul : (prod hreal hreal) -> (prod hreal hreal) -> prod hreal hreal := fun _23757 : prod hreal hreal => fun _23758 : prod hreal hreal => @pair hreal hreal (hreal_add (hreal_mul (@fst hreal hreal _23757) (@fst hreal hreal _23758)) (hreal_mul (@snd hreal hreal _23757) (@snd hreal hreal _23758))) (hreal_add (hreal_mul (@fst hreal hreal _23757) (@snd hreal hreal _23758)) (hreal_mul (@snd hreal hreal _23757) (@fst hreal hreal _23758))).

Definition treal_le : (prod hreal hreal) -> (prod hreal hreal) -> Prop := fun _23779 : prod hreal hreal => fun _23780 : prod hreal hreal => hreal_le (hreal_add (@fst hreal hreal _23779) (@snd hreal hreal _23780)) (hreal_add (@fst hreal hreal _23780) (@snd hreal hreal _23779)).

(*Lemma treal_le_refl x : treal_le x x.
Proof.
  unfold treal_le. destruct x as [x1 x2]. simpl. apply hreal_le_refl.
Qed.

Add Relation _ treal_le
    reflexivity proved by treal_le_refl
    (*transitivity proved by treal_le_trans*)
as treal_le_rel.*)

Definition treal_inv : (prod hreal hreal) -> prod hreal hreal := fun _23801 : prod hreal hreal => @COND (prod hreal hreal) ((@fst hreal hreal _23801) = (@snd hreal hreal _23801)) (@pair hreal hreal (hreal_of_num (NUMERAL N0)) (hreal_of_num (NUMERAL N0))) (@COND (prod hreal hreal) (hreal_le (@snd hreal hreal _23801) (@fst hreal hreal _23801)) (@pair hreal hreal (hreal_inv (@ε hreal (fun d : hreal => (@fst hreal hreal _23801) = (hreal_add (@snd hreal hreal _23801) d)))) (hreal_of_num (NUMERAL N0))) (@pair hreal hreal (hreal_of_num (NUMERAL N0)) (hreal_inv (@ε hreal (fun d : hreal => (@snd hreal hreal _23801) = (hreal_add (@fst hreal hreal _23801) d)))))).

Definition treal_eq : (prod hreal hreal) -> (prod hreal hreal) -> Prop := fun _23810 : prod hreal hreal => fun _23811 : prod hreal hreal => (hreal_add (@fst hreal hreal _23810) (@snd hreal hreal _23811)) = (hreal_add (@fst hreal hreal _23811) (@snd hreal hreal _23810)).

Lemma treal_eq_refl x : treal_eq x x.
Proof. reflexivity. Qed.

Lemma treal_eq_sym x y : treal_eq x y -> treal_eq y x.
Proof.
  unfold treal_eq. destruct x as [x1 x2]; destruct y as [y1 y2]; simpl.
  intro e. symmetry. exact e.
Qed.

Lemma treal_eq_trans x y z : treal_eq x y -> treal_eq y z -> treal_eq x z.
Proof.
  unfold treal_eq.
  destruct x as [x1 x2]; destruct y as [y1 y2]; destruct z as [z1 z2]; simpl.
  intros xy yz.
  assert (h: hreal_add (hreal_add x1 z2) (hreal_add y1 y2)
             = hreal_add (hreal_add z1 x2) (hreal_add y1 y2)).
  rewrite hreal_add_assoc. rewrite <- (hreal_add_assoc z2).
  rewrite (hreal_add_sym _ y2). rewrite <- hreal_add_assoc.
  rewrite (hreal_add_sym z2). rewrite xy, yz.

  rewrite hreal_add_assoc. rewrite (hreal_add_sym (hreal_add z1 x2)).
  rewrite hreal_add_assoc. rewrite (hreal_add_sym y2).
  rewrite (hreal_add_sym z1 x2). rewrite hreal_add_assoc.
  reflexivity. apply hreal_add_lcancel in h. exact h.
Qed.

Add Relation _ treal_eq
    reflexivity proved by treal_eq_refl
    symmetry proved by treal_eq_sym
    transitivity proved by treal_eq_trans
as treal_eq_rel.

Add Morphism treal_add
    with signature treal_eq ==> treal_eq ==> treal_eq
      as treal_add_morph.
Proof.
  intros f f' ff' g g' gg'. unfold treal_eq, treal_add; simpl.
  unfold treal_eq in ff', gg'.
  destruct f as [x1 x2]; destruct f' as [x'1 x'2];
    destruct g as [y1 y2]; destruct g' as [y'1 y'2]; simpl in *.
  rewrite (hreal_add_sym x1). rewrite hreal_add_assoc.
  rewrite <- (hreal_add_assoc x1). rewrite ff'.
  rewrite (hreal_add_sym x2). rewrite (hreal_add_assoc x'1 y'1).
  rewrite <- (hreal_add_assoc y'1). rewrite <- gg'.
  rewrite (hreal_add_assoc y1). rewrite (hreal_add_sym y'2).
  rewrite <- (hreal_add_assoc x'1). rewrite (hreal_add_sym x'1 y1).
  rewrite !hreal_add_assoc. reflexivity.
Qed.

(*Add Morphism treal_le
    with signature treal_eq ==> treal_eq ==> iff
      as treal_le_morph.
Proof.
Abort.*)

(*****************************************************************************)
(* HOL-Light definition of real numbers. *)
(*****************************************************************************)

Definition real := quotient treal_eq.

Definition mk_real := mk_quotient treal_eq.
Definition dest_real := dest_quotient treal_eq.

Lemma axiom_23 : forall (a : real), (mk_real (dest_real a)) = a.
Proof. exact (mk_dest_quotient treal_eq). Qed.

Lemma axiom_24_aux : forall r, (exists x, r = treal_eq x) -> dest_real (mk_real r) = r.
Proof. exact (dest_mk_aux_quotient treal_eq). Qed.

Lemma axiom_24 : forall (r : (prod hreal hreal) -> Prop), ((fun s : (prod hreal hreal) -> Prop => exists x : prod hreal hreal, s = (treal_eq x)) r) = ((dest_real (mk_real r)) = r).
Proof. exact (dest_mk_quotient treal_eq). Qed.
