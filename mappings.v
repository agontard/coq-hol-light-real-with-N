Require Import mathcomp.classical.classical_sets.
Import ssreflect ssrnat ssrfun eqtype choice ssrbool boolp HB.structures.
Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

(****************************************************************************)
(* Rocq theory for encoding HOL-Light proofs. *)
(****************************************************************************)

Lemma ext_fun {A B} {f g : A -> B} : f = g -> forall x, f x = g x.
Proof. intros fg x. rewrite fg. reflexivity. Qed.

Lemma eq_false_negb_true b : b = false -> negb b = true.
Proof. intro e. subst. reflexivity. Qed.

(****************************************************************************)
(* Type of non-empty types, used to interpret HOL-Light types types. *)
(****************************************************************************)

Require Export HOLLight_Real_With_N.type.

(* The most important types and type constructors already have an instance as Type' *)

(****************************************************************************)
(* Extensionalities *)
(****************************************************************************)

(* tactic ext should cover almost all cases of need with functional and propositional
   extensionality and their combinations. *)

(* Axiom propext : forall P Q, P <-> Q -> P = Q *)
Lemma prop_ext : forall P Q : Prop, (P -> Q) -> (Q -> P) -> P = Q.
Proof.
  by move=> *; eqProp.
Qed.

(* Axiom functional_extensionality_dep :
   forall f g, (forall x, f x = g x) -> f = g *)
Notation funext := functional_extensionality_dep.

(* applies them to all arguments and propositions at once *)
Tactic Notation "ext" :=
  let rec ext' := (let H := fresh in
    first [apply funext | eqProp]=> H ; try ext' ; move:H)
  in ext'.

(* with a counter added to the context to apply it for exactly n arguments/propositions *)
Variant internal_witness : forall A, A -> Type :=
  w0 A : forall a : A, internal_witness a.

Definition addone : forall n, internal_witness n -> internal_witness n.+1 :=
  fun n _ => w0 n.+1.

Ltac typecheck A a := assert_succeeds let s:=fresh in set (s := a : A).

Tactic Notation "ext" integer(n) :=
  let ext0 x w := (first [apply funext | eqProp]=>x ; set (w := w0 x))
   (* choosing the fresh variables inside ext0 fails to create new variable names. *)
  in do n (let x := fresh in let w := fresh in ext0 x w) ;
  repeat match goal with | x : _ |- _  =>
  lazymatch goal with w : internal_witness x |- _ => clear w ; revert x end end.

(* Example of use :
Lemma test : (fun f g : nat -> nat => (f=g)) = (fun f g : nat -> nat => (f=g)).
Proof.
(*  ext=> f g H n. *) (* or *)
(*  ext 3 => f g H. *)
  auto.
Qed. *)

(* Small issue : the witnesses are present in the proof term. How bad is it?
   Example :
   Lemma test (f : nat -> nat -> nat -> nat -> nat -> nat -> nat -> nat) : f = f.
   Proof. by ext 7. Qed.
   Print test.

   This means that ext should be prefered to ext n if possible. *)

Tactic Notation "gen_intro" constr(h) simple_intropattern(e) :=
  generalize h; clear e; intros e.

Tactic Notation "intro_ext" simple_intropattern(e) :=
  intros e; gen_intro (ext_fun e) e.

Tactic Notation "intro_ext" simple_intropattern(e) ident(x) :=
  intro_ext e; gen_intro (ext_fun (e x)) e.

(****************************************************************************)
(* Repeating exists. *)
(****************************************************************************)

Tactic Notation "exist" uconstr(x1) uconstr(x2) :=
  exists x1 ; exists x2.

Tactic Notation "exist" uconstr(x1) uconstr(x2) uconstr(x3) :=
  exists x1 ; exists x2 ; exists x3.

Tactic Notation "exist" uconstr(x1) uconstr(x2) uconstr(x3) uconstr(x4) :=
  exists x1 ; exists x2 ; exists x3 ; exists x4.

Tactic Notation "exist" uconstr(x1) uconstr(x2) uconstr(x3) uconstr(x4) uconstr(x5) :=
  exists x1 ; exists x2 ; exists x3 ; exists x4 ; exists x5.

(****************************************************************************)
(* Coercion from Prop to bool? *)
(****************************************************************************)

Coercion asbool : Sortclass >-> bool.

Ltac booleqsimp := rewrite ?eqb_id ?eqbF_neg.

Lemma bool_eq_decompP : forall a b : bool, is_true (a==b) = (is_true a = is_true b).
Proof.
  by move=> [] [] ; ext ; booleqsimp ; rewrite //= ; first (move <-) ; last (move ->).
Qed.

Ltac AllProp := rewrite -?eq_opE ?bool_eq_decompP ?asboolE.

(* Check and : bool -> bool -> bool. *)

(****************************************************************************)
(* Coercion from nat to N ? *)
(****************************************************************************)
(* TODO : For another time, if the double coercion between bool and Prop works out.
          it would require to either remove nat_of_bin as a coercion and implement
          N.of_nat and N.to_nat from Stdlib.NArith.Nnat as coercions instead,
          or prove isomorphism (conversions for the most important functions
          and relations) as ssrnat has basically none, especially for bin_of_nat.
          could allow to use for example nth or length on lists.

Require Import Stdlib.NArith.BinNat.
Coercion bin_of_nat : nat >-> N.

(* nat_of_bin is already defined as a coercion in ssrnat. *)

(* Check S : N -> N. *) *)

(****************************************************************************)
(* Curryfied versions of some Rocq connectives. *)
(****************************************************************************)

Definition arr (A B : Type') : Type' := A -> B.

Definition imp (P Q : Prop) : Prop := P -> Q.

Definition ex1 : forall (A : Type'), (A -> Prop) -> Prop := fun A P => exists! x, P x.

(****************************************************************************)
(* Proof of some HOL-Light rules. *)
(****************************************************************************)

Lemma MK_COMB (a b : Type') (s t : a -> b) (u v : a) (h1 : s = t) (h2 : u = v)
  : (s u) = (t v).
Proof. by rewrite h1 h2. Qed.

Lemma EQ_MP (p q : Prop) (e : p = q) (h : p) : q.
Proof. by rewrite -e. Qed.

(* erefl with non-implicit second parameter. *)
Definition REFL A x := @erefl A x.

(****************************************************************************)
(* Proof of some natural deduction rules. *)
(****************************************************************************)

Lemma or_intro1 (p:Prop) (h : p) (q:Prop) : p \/ q.
Proof. exact (@or_introl p q h). Qed.

Lemma or_elim (p q : Prop) (h : p \/ q) (r : Prop) (h1: p -> r) (h2: q -> r) : r.
Proof. exact (@or_ind p q r h1 h2 h). Qed.

Lemma ex_elim a (p : a -> Prop)
  (h1 : exists x, p x) (r : Prop) (h2 : forall x : a, (p x) -> r) : r.
Proof. exact (@ex_ind a p r h2 h1). Qed.

(****************************************************************************)
(* Hilbert's ε operator. *)
(****************************************************************************)

Definition ε (A : Type') (P : A -> Prop) := get P.

Definition ε_spec {A : Type'} {P : A -> Prop} : (exists x, P x) -> P (ε P) := @getPex _ P.

Lemma align_ε (A : Type') (P : A -> Prop) a : P a -> (forall x, P a -> P x -> a = x) -> a = ε P.
Proof. 
  by move => ha ; apply ; last (apply ε_spec ; exists a).
Qed.

(* The definition of an HOL-Light function that is recursively defined
on some inductive type usually looks like:

  [ε (fun g => forall uv, P (g uv)) uv0]

  where P does not depend on uv (unused variable). *)

(* [gobble f uv] replaces the occurrences of [f uv] by a new symbol
named [f] too and removes [uv] from the context, assuming that [f uv]
does not actually depends on [uv]. *)
Ltac gobble f uv :=
  let g := fresh in
  set (g := f uv) in * ;
  clearbody g ; simpl in g ;
  clear f uv ; rename g into f.

(* From a goal of the form [a = ε (fun a' => forall uv, P (a' uv)) uv0],
align_ε generates two subgoals [P a] and [forall x, P a -> P x -> a = x]. *)
Ltac align_ε :=
  let rec aux :=
    lazymatch goal with
    | |- _ ?x = ε _ ?x => apply (f_equal (fun f => f x)) ; aux
    | |- ?a = ε _ ?r =>
        (* Replace the goal by (fun _ => a = ε ?P) *)
        apply (f_equal (fun g => g r) (x := fun _ => a)) ;
        aux ;
        [ let uv := fresh in
          intro uv ; clear uv 

        | let a' := fresh in
          let uv := fresh in
          let H' := fresh in
          let H := fresh in
          intros a' H H' ; ext 1=> uv ;
          specialize (H uv) ; (* As [P] starts with [forall uv] *)
          specialize (H' uv) ;
          simpl ((fun _ => a) uv) in * ; (* Simplifies to [a] so that [uv] only appears in [a' uv] *)
          gobble a' uv ;
          revert a' H H' (* Revert [a'], [P a] and [P a'] to reuse them in other tactics *)
        ]
    | |- ?a = ε ?P => apply align_ε (* Replaces the goal [a = ε P] with two goals [P a] and
                                       [forall x, P a => P x => x = a]. *)
    end
  in aux.

Lemma is_True P : (P = True) = P.
Proof.
  by ext=> // ->.
Qed.

(* From a proof of P (either a hypothesis or a lemma), rewrite P into True. *)
Ltac is_True H :=
  let H' := fresh in set (H' := H) ;
  match type of H' with ?P => rewrite <- (is_True P) in H' ;
    rewrite -> H' in * ; clear H' ; try clear H end.

Lemma is_False P : (P = False) = ~ P.
Proof.
  by ext=> // ->.
Qed.

(* From hypothesis H : ~P, rewrite P into False *)
Ltac is_False H :=
  let H' := fresh in set (H' := H) ;
  match type of H' with ~?P => rewrite <- (is_False P) in H' ;
    rewrite -> H' in * ; clear H' ; try clear H end.

Lemma prop_degen : forall P, P = True \/ P = False.
Proof.
  move=>P ; rewrite is_False is_True ; exact (EM _).
Qed.

Lemma sym {A} (x y : A) : (x = y) = (y = x).
Proof. by ext. Qed.

(****************************************************************************)
(* if then else over Prop *)
(****************************************************************************)

Definition COND (A : Type) (P : Prop) (x y : A) := if P then x else y.

Lemma if_True (A : Type) (x y : A) : (if True then x else y) = x.
Proof.
  by rewrite/asbool ; case pselect.
Qed.

Lemma if_False (A : Type) (x y : A) : (if False then x else y) = y.
Proof.
by rewrite/asbool ; case pselect.
Qed.

Lemma COND_def {A : Type'} : (@COND A) = (fun t : Prop => fun t1 : A => fun t2 : A => @ε A (fun x : A => ((t = True) -> x = t1) /\ ((t = False) -> x = t2))).
Proof.
  ext=> P x y ; align_ε ; first by split=> -> ; rewrite/COND ?if_True ?if_False.
  move=> f' [CT CF] [f'T f'F] ; case : (prop_degen P) => H ; first by rewrite (CT H) f'T.
  by rewrite (CF H) f'F.
Qed.

(* The following are useful tools to work with COND :
   - Tactic if_triv replaces [if P then x else y] in the goal with either x or y
     assuming P or ~P is derivable with easy.
   - Tactic if_intro transforms a goal [P (if Q then x else y)]
     into two goals [Q -> P x] and [~Q -> P y] and
     simplifies all other [if Q then x' else y'] even if x<>x' or y<>y'
   - Lemma if_elim destructs hypothesis [if P then Q else R]
     as if it were (P /\ Q) \/ (~P /\ R) *)

Lemma if_triv_True (A : Type') (P : Prop) (x y : A) : P -> (if P then x else y) = x.
Proof.
  by rewrite -{1}(is_True P) => -> ; exact (if_True _ _).
Qed.

Lemma if_triv_False (A : Type') (P : Prop) (x y : A) : ~P -> (if P then x else y) = y.
Proof.
  by rewrite -{1}(is_False P) => -> ; exact (if_False _ _).
Qed.

Tactic Notation "if_triv" :=
  rewrite/COND ;
  (rewrite if_triv_True ; [auto | easy]) +
  (rewrite if_triv_False ; [auto | easy]).

(* If needed, specify which P is trivial *)
Tactic Notation "if_triv" constr(P) :=
  rewrite/COND ;
  (rewrite (if_triv_True _ P) ; [auto | easy]) +
  (rewrite (if_triv_False _ P) ; [auto | easy]).

Tactic Notation "if_triv" "using" constr(H) :=
  rewrite/COND ; let H' := fresh in set (H' := H) ;
  (rewrite if_triv_True ; [auto | now auto using H']) +
  (rewrite if_triv_False ; [auto | now auto using H']) ; clear H'.

Tactic Notation "if_triv" constr(P) "using" constr(H) :=
  rewrite/COND ; let H' := fresh in set (H' := H) ;
  (rewrite (if_triv_True _ P) ; [auto | now auto using H']) +
  (rewrite (if_triv_False _ P) ; [auto | now auto using H']) ; clear H'.

Lemma if_intro (A : Type') (Q : Prop) (P : A -> Prop) (x y : A) :
  (Q -> P x) -> (~Q -> P y) -> P (if Q then x else y).
Proof. by case (pselect Q)=>H ; if_triv. Qed.

(* applies if_intro then also clears all if with
   the same hypothesis *)

Ltac if_intro :=
  rewrite/COND ;
  let H := fresh in
  apply if_intro => H ;
  [ repeat rewrite (if_triv_True _ _ H)
  | repeat rewrite (if_triv_False _ _ H)] ;
  move : H.

Lemma if_elim (P Q R G : Prop) : (if P then Q else R) -> (P -> Q -> G) -> (~ P -> R -> G) -> G.
Proof.
  by case (pselect P) => H ; if_triv.
Qed.

Definition COND_dep (Q: Prop) (C: Type) (f1: Q -> C) (f2: ~Q -> C) : C :=
  match pselect Q with
  | left x => f1 x
  | right x => f2 x
  end.

(****************************************************************************)
(* Alignment of partial functions *)
(****************************************************************************)

(* Whenever functions defined in HOL-Light are only defined in some specific cases
   with ε, it can be defined in Rocq with an if then else, to have a meaningful value
   for these specific cases and default to the default ε-chosen function's
   value elsewhere. For example, functions defined on finite sets.

   align_ε_if works similarly to align_ε' in this case, with restraining
   uniqueness to the relevant cases for the function *)

Lemma align_ε_if1 {A B : Type'}
  (Q : A -> Prop) (f : A -> B) (P : (A -> B) -> Prop) :
  P f ->
  (forall f', P f -> P f' -> forall x, Q x -> f x = f' x) ->
  forall x, (if Q x then f x else ε P x) = ε P x.
Proof.
  move=> Hf Hunique x ; if_intro => // ; apply Hunique => //.
  by apply ε_spec ; exists f.
Qed.

Lemma align_ε_if2 {A B C : Type'}
  (Q : (A -> B -> Prop)) (f : A -> B -> C) (P : (A -> B -> C) -> Prop) :
  P f ->
  (forall f', P f -> P f' -> forall x y, Q x y -> f x y = f' x y) ->
  forall x y, (if Q x y then f x y else ε P x y) = ε P x y.
Proof.
  move=> Hf Hunique x y ; if_intro => // ; apply Hunique => //.
  by apply ε_spec ; exists f.
Qed.

Lemma align_ε_if3 {A B C D : Type'}
  (Q : (A -> B -> C -> Prop)) (f : A -> B -> C -> D) (P : (A -> B -> C -> D) -> Prop) :
  P f ->
  (forall f', P f -> P f' -> forall x y z, Q x y z -> f x y z = f' x y z) ->
  forall x y z, (if Q x y z then f x y z else ε P x y z) = ε P x y z.
Proof.
  move=> Hf Hunique x y z; if_intro => // ; apply Hunique => //.
  by apply ε_spec ; exists f.
Qed.

Arguments align_ε_if1 {_ _} _ _.
Arguments align_ε_if2 {_ _ _} _ _.
Arguments align_ε_if3 {_ _ _ _} _ _.

Ltac funext :=
  let rec funext' := (let x := fresh "x" in
    apply funext => x ; try funext' ; move:x)
  in funext'.

Ltac align_ε_if :=
  let rec aux :=
    lazymatch goal with
    | |- ?f = ε _ ?uv => generalize uv ; aux ;
          [ intros _
          | let f' := fresh in
            let uv := fresh in
            let Hf := fresh in
            let Hf' := fresh in
            let x := fresh "x" in
            let y := fresh "y" in
            let HP := fresh in
            try intros f' Hf Hf' uv x y HP ; try intros f' Hf Hf' uv x HP ;
            specialize (Hf uv) ;
            specialize (Hf' uv) ;
            simpl (_ uv) in * ;
            gobble f' uv ;
            revert HP ; try revert y ; revert f' Hf Hf' x ]
    | |- _ = ε _ => funext ; (* helping with pattern matching by
                                universally quantifying on the arguments *)
      lazymatch goal with
      | |- forall x, (if `[<?P>] then ?f else _) = _ =>
             apply (align_ε_if1 (fun x => P) (fun x => f))
      | |- forall x y, (if `[<?P>] then ?f else _) = _ =>
             apply (align_ε_if2 (fun x y => P) (fun x y => f))
      | |- forall x y z, (if `[<?P>] then ?f else _) = _ =>
             apply (align_ε_if3 (fun x y z => P) (fun x y z => f)) end
    | |- forall uv, _ = ε _ uv => intro uv ; try funext ; revert uv ;
      lazymatch goal with
      | |- forall x, (if `[<?P>] then ?f else _) = _ =>
             apply (align_ε_if1 (fun x => P) (fun x => f))
      | |- forall x y, (if `[<?P>] then ?f else _) = _ =>
             apply (align_ε_if2 (fun x y => P) (fun x y => f))
      | |- forall x y z, (if `[<?P>] then ?f else _) = _ =>
             apply (align_ε_if3 (fun x y z => P) (fun x y z => f)) end end
  in aux.

(****************************************************************************)
(* Miscellaneous. *)
(****************************************************************************)

Lemma GABS_def {A : Type'} : (@ε A) = (fun P : A -> Prop => @ε A P).
Proof. exact erefl. Qed.

Lemma GEQ_def {A : Type'} : (@eq A) = (fun a : A => fun b : A => a = b).
Proof. exact erefl. Qed.

Lemma _UNGUARDED_PATTERN_def : and = (fun p : Prop => fun r : Prop => p /\ r).
Proof. exact erefl. Qed.

Lemma _FALSITY__def : False = False.
Proof. exact erefl. Qed.

Lemma hashek_def : True = True.
Proof. exact erefl. Qed.

Definition cancel2 (A B : Type) (f : A -> B) (g : B -> A) := cancel g f /\ cancel f g.

Lemma ISO_def {A B : Type'} : (@cancel2 A B) = (fun _17569 : A -> B => fun _17570 : B -> A => (forall x : B, (_17569 (_17570 x)) = x) /\ (forall y : A, (_17570 (_17569 y)) = y)).
Proof. ext 2=>* ; exact erefl. Qed.

(****************************************************************************)
(* Proof of HOL-Light axioms. *)
(****************************************************************************)

Lemma axiom_0 : forall {A B : Type'}, forall t : A -> B, (fun x : A => t x) = t.
Proof. by []. Qed.

Lemma axiom_1 : forall {A : Type'}, forall P : A -> Prop, forall x : A, (P x) -> P (@ε A P).
Proof.
  by move=> A P x H ; apply ε_spec ; exists x.
Qed.

(****************************************************************************)
(* Alignment of connectives. *)
(****************************************************************************)

Lemma ex1_def : forall {A : Type'}, (@ex1 A) = (fun P : A -> Prop => (ex P) /\ (forall x : A, forall y : A, ((P x) /\ (P y)) -> x = y)).
Proof.
  rewrite/ex1=> A ; ext=> [P [x [Hx Hunique]] | P [[x Hx] Hunique]].
  - by split ; first exists x ; last move=> y z [/Hunique<- /Hunique<-].
  - by exists x ; split ; last (move=> * ; apply Hunique).
Qed.

Lemma F_def : False = (forall p : Prop, p).
Proof.
  by eqProp=>H ; first destruct H ; apply H.
Qed.

Lemma not_def : not = (fun p : Prop => p -> False).
Proof. reflexivity. Qed.

Lemma or_def : or = (fun p : Prop => fun q : Prop => forall r : Prop, (p -> r) -> (q -> r) -> r).
Proof.
  ext=> [P Q []|P Q] ; last apply ; tauto.
Qed.

Lemma ex_def : forall {A : Type'}, (@ex A) = (fun P : A -> Prop => forall q : Prop, (forall x : A, (P x) -> q) -> q).
Proof.
  by move=> A ; ext=> P ; last apply ; firstorder.
Qed.

Lemma all_def : forall {A : Type'}, (@all A) = (fun P : A -> Prop => P = (fun x : A => True)).
Proof.
  by move=>A ; ext=> [|| P ->].
Qed.

Lemma imp_def : imp = (fun p : Prop => fun q : Prop => (p /\ q) = p).
Proof.
  by ext=>[||P Q <-] ; firstorder.
Qed.

Lemma and_def : and = (fun p : Prop => fun q : Prop => (fun f : Prop -> Prop -> Prop => f p q) = (fun f : Prop -> Prop -> Prop => f True True)).
Proof.
  ext=>[||P Q e] ; last by pattern and ; rewrite e.
  1,2 : by move=> P Q [HP HQ] ; is_True HP ; is_True HQ.
Qed.

Lemma T_def : True = ((fun p : Prop => p) = (fun p : Prop => p)).
Proof. by ext 1. Qed.

(*****************************************************************************)
(* Alignment of subtypes. *)
(*****************************************************************************)

Require Import Stdlib.Logic.ProofIrrelevance.

Section Subtype.

  Variables (A : Type) (P : A -> Prop) (a : A) (h : P a).

  Definition subtype (h : P a) := {x | P x}.

  HB.instance Definition _ := HOL_isPointed.Build (subtype h) (exist P a h).

  Definition dest : subtype h -> A := fun x => proj1_sig x.

  Definition mk : A -> subtype h :=
    fun x => COND_dep (exist P x) (fun _ => exist P a h).

  Lemma dest_mk_aux x : P x -> (dest (mk x) = x).
  Proof.
    by move=> Px ; rewrite/mk/COND_dep ; case:(pselect (P x)).
  Qed.

  Lemma dest_mk x : P x = (dest (mk x) = x).
  Proof.
    by ext=> [|<-] ; first exact (@dest_mk_aux x) ; last case:(mk x).
  Qed.

  Lemma mk_dest x : mk (dest x) = x.
  Proof.
    rewrite/mk/COND_dep; case: (pselect (P (dest x))); case x ; last by [].
    by move=>a' p pa' ; rewrite (proof_irrelevance _ p pa').
  Qed.

  Lemma dest_inj x y : dest x = dest y -> x = y.
  Proof.
    intro e. destruct x as [x i]. destruct y as [y j]. simpl in e.
    subst y. rewrite (proof_irrelevance _ i j). reflexivity.
  Qed.

  Lemma mk_inj x y : P x -> P y -> mk x = mk y -> x = y.
  Proof.
    rewrite/mk/COND_dep=>hx hy.
    by case:(pselect (P x)); case (pselect (P y))=> Hx Hy ; try move=>[=].
  Qed.

End Subtype.

Arguments dest [_] [_] [_] _.
Arguments mk_dest [_] [_] [_] _.

(*****************************************************************************)
(* Alignment of quotients. *)
(*****************************************************************************)

Section Quotient.

  Variables (A : Type') (R : A -> A -> Prop).

  Definition is_eq_class X := exists a, X = R a.

  Definition class_of x := R x.

  Lemma is_eq_class_of x : is_eq_class (class_of x).
  Proof. exists x. reflexivity. Qed.

  Lemma non_empty : is_eq_class (class_of point).
  Proof. exists point. reflexivity. Qed.

  Definition quotient := subtype non_empty.

  Definition mk_quotient : (A -> Prop) -> quotient := mk non_empty.
  Definition dest_quotient : quotient -> (A -> Prop) := dest non_empty.

  Lemma mk_dest_quotient : forall x, mk_quotient (dest_quotient x) = x.
  Proof. exact (mk_dest non_empty). Qed.

  Lemma dest_mk_aux_quotient : forall x, is_eq_class x -> (dest_quotient (mk_quotient x) = x).
  Proof. exact (dest_mk_aux non_empty). Qed.

  Lemma dest_mk_quotient : forall x, is_eq_class x = (dest_quotient (mk_quotient x) = x).
  Proof. exact (dest_mk non_empty). Qed.

  Definition elt_of : quotient -> A := fun x => ε (dest_quotient x).

  Variable R_refl : forall a, R a a.

  Lemma eq_elt_of a : R a (ε (R a)).
  Proof. apply ε_spec. exists a. apply R_refl. Qed.

  Lemma dest_quotient_elt_of x : dest_quotient x (elt_of x).
  Proof.
    unfold elt_of, dest_quotient, dest. destruct x as [c [a h]]; simpl. subst c.
    apply ε_spec. exists a. apply R_refl.
  Qed.

  Variable R_sym : forall x y, R x y -> R y x.
  Variable R_trans : forall x y z, R x y -> R y z -> R x z.

  Lemma dest_quotient_elim x y : dest_quotient x y -> R (elt_of x) y.
  Proof.
    unfold elt_of, dest_quotient, dest. destruct x as [c [a h]]; simpl. subst c.
    intro h. apply R_trans with a. apply R_sym. apply eq_elt_of. exact h.
  Qed.

  Lemma eq_class_intro_elt (x y: quotient) : R (elt_of x) (elt_of y) -> x = y.
  Proof.
    destruct x as [c [x h]]. destruct y as [d [y i]]. unfold elt_of. simpl.
    intro r. apply subset_eq_compat. subst c. subst d.
    ext=> a j.

    apply R_trans with (ε (R y)). apply eq_elt_of.
    apply R_trans with (ε (R x)). apply R_sym. apply r.
    apply R_trans with x. apply R_sym. apply eq_elt_of. exact j.

    apply R_trans with (ε (R x)). apply eq_elt_of.
    apply R_trans with (ε (R y)). apply r.
    apply R_trans with y. apply R_sym. apply eq_elt_of. exact j.
  Qed.

  Lemma eq_class_intro (x y: A) : R x y -> R x = R y.
  Proof.
    intro xy. ext=> a h.
    apply R_trans with x. apply R_sym. exact xy. exact h.
    apply R_trans with y. exact xy. exact h.
  Qed.

  Lemma eq_class_elim (x y: A) : R x = R y -> R x y.
  Proof.
    intro h. generalize (ext_fun h y); intro hy.
    is_True (R_refl y). now rewrite hy.
  Qed.

  Lemma mk_quotient_elt_of x : mk_quotient (R (elt_of x)) = x.
  Proof.
    apply eq_class_intro_elt. set (a := elt_of x). unfold elt_of.
    rewrite dest_mk_aux_quotient. apply R_sym. apply eq_elt_of.
    exists a. reflexivity.
  Qed.

End Quotient.
Arguments dest_quotient [_] _.
Arguments mk_dest_quotient [_] _.
Arguments dest_mk_aux_quotient [_] _ _ _.

(****************************************************************************)
(* Alignment of inductive propositions. *)
(****************************************************************************)

Ltac breakgoal :=
  let rec breakgoal' :=
    match reverse goal with
    | |- _ \/ _ => left + right ; breakgoal' (* Try both *)
    | x : ?T |- exists _ : ?T, _ => (* When the witness is already in the context.
                                       Doesn't try a witness for multiple quantifers
                                       by leaving a trace [H : x = x] whenever trying
                                       [exists x].
                                       Going from the top down with "reverse goal"
                                       because they should appear in order. *)
      match goal with
      | H : x = x |- _ => fail
      | _ => let H := fresh in
        assert (H : x = x) ; try reflexivity ; exists x ; breakgoal' end
    | |- _ /\ _ => repeat split ; auto ; fail
         (* if auto cannot do the job, the tactic should fail to branch back.  *)
    | |- _ => auto ; fail
    end
  in breakgoal'.

(* simply decomposing each hypothesis that we might encounter,
   a lot faster than going brutally with firstorder *)
Ltac full_destruct := repeat match goal with
  | H : _ /\ _ |- _ => destruct H
  | H : exists x, _ |- _ => destruct H
  | H : _ \/ _ |- _ => destruct H end.

(* In HOL_Light, an inductive defintion is top-down :
   if [Case_i x1 ... xn : Hyps_i x1 ... xn -> P (f_i x1 ... xn)] for 1 <= i <= k
   are the constructors / rules of P, then :
   [P x = forall P', (forall x', Case_1' x' \/ ... \/ Case_k' x' -> P' x') -> P' x]
   where [Case_i' x' := exists x1 ... xn, f_i x1 ... xn = x' /\ Hyps_i x1 ... xn]

   Let P_h x := Forall P', H' -> P' x' denote the HOL_Light definition of P
   and P_r the Rocq Inductive definition.
   *)
Ltac ind_align :=
  let x := fresh "x" in
  let y := fresh "y" in
  let z := fresh "z" in
  let H := fresh in
  try ext=> x y z H ; try ext => x y H ; try ext => x H ; try ext => H ;
  (* Prove equality by double implication *)
  [ let P' := fresh "P'" in
    let H' := fresh "H'" in (* Proving [P_r x -> P_h x] *)
    intros P' H' ; induction H ; apply H' ;
    (* Induction on hypothesis [P_r x] replaces [x] according to Case_i for each i.
       to prove [P' x] we apply [H']. *)
    try breakgoal (* Trying to automatically find and solve Case_i'.
                     The Hyps_i are in the context. *)
  | (* Proving [P_h x -> P_r x] *)
    apply H ; (* Replaces goal [P_r x] with [H'] *)
    clear H ; try clear x ; try clear y ; try clear z ; (* H' talks about fresh variables *)
    try intros x y z H ; try intros x y H ; try intros x H ;
    full_destruct ; (* Destructing H results in one goal per case, and separates the hypotheses *)
    repeat match goal with
    H : _ |- _ => rewrite H (* not much to do, each clause should be proved with a rule,
                               we just try to rewrite [a = f x1 ... xn] if it exists *)
    end ; try now (constructor;auto) ].

(****************************************************************************)
(* Alignment of the unit type. *)
(****************************************************************************)

Definition one_ABS : Prop -> unit := fun _ => tt.

Definition one_REP : unit -> Prop := fun _ => True.

Lemma axiom_2 : forall (a : unit), (one_ABS (one_REP a)) = a.
Proof. intro a. destruct a. reflexivity. Qed.

Lemma axiom_3 : forall (r : Prop), ((fun b : Prop => b) r) = ((one_REP (one_ABS r)) = r).
Proof. intro r. compute. rewrite (sym True r) is_True. reflexivity. Qed.

Lemma one_def : tt = ε one_REP.
Proof. by case:(ε one_REP). Qed.

(****************************************************************************)
(* Alignment of the product type constructor. *)
(****************************************************************************)

Definition mk_pair {A B : Type'} :=
  fun x : A => fun y : B => fun a : A => fun b : B => (a = x) /\ (b = y).

Lemma mk_pair_def {A B : Type'} : (@mk_pair A B) = (fun x : A => fun y : B => fun a : A => fun b : B => (a = x) /\ (b = y)).
Proof. exact erefl. Qed.

Lemma mk_pair_inj {A B : Type'} {x x' : A} {y y' : B} :
  mk_pair x y = mk_pair x' y' -> x = x' /\ y = y'.
Proof.
  intro_ext e x.
  gen_intro (e y) e.
  unfold mk_pair in e.
  by rewrite -e.
Qed.

Definition ABS_prod : forall {A B : Type'}, (A -> B -> Prop) -> prod A B :=
  fun A B f => ε (fun p => f = mk_pair (fst p) (snd p)).

Lemma ABS_prod_mk_pair {A B : Type'} {x : A} {y : B} :
  (x,y) = ABS_prod (mk_pair x y).
Proof.
  unfold ABS_prod.
  align_ε.
  - reflexivity.
  - move=> [x' y'] _ h.
    apply pair_equal_spec.
    exact (mk_pair_inj h).
Qed.

Definition REP_prod : forall {A B : Type'}, (prod A B) -> A -> B -> Prop :=
  fun A B p a b => mk_pair (fst p) (snd p) a b.

Lemma pair_def {A B : Type'} : (@pair A B) = (fun x : A => fun y : B => @ABS_prod A B (@mk_pair A B x y)).
Proof.
  ext=> x y.
  exact ABS_prod_mk_pair.
Qed.

Lemma FST_def {A B : Type'} : (@fst A B) = (fun p : prod A B => @ε A (fun x : A => exists y : B, p = (@pair A B x y))).
Proof.
  ext=> [[x y]].
  align_ε.
  - exists y.
    reflexivity.
  - by move => x' _ [y' ->].
Qed.

Lemma SND_def {A B : Type'} : (@snd A B) = (fun p : prod A B => @ε B (fun y : B => exists x : A, p = (@pair A B x y))).
Proof.
  ext=> [[x y]].
  align_ε.
  - exists x.
    reflexivity.
  - by move=> y' _ [x' ->].
Qed.

Lemma axiom_4 : forall {A B : Type'} (a : prod A B), (@ABS_prod A B (@REP_prod A B a)) = a.
Proof. intros A B [a b]. symmetry. exact ABS_prod_mk_pair. Qed.

Lemma axiom_5 : forall {A B : Type'} (r : A -> B -> Prop), ((fun x : A -> B -> Prop => exists a : A, exists b : B, x = (@mk_pair A B a b)) r) = ((@REP_prod A B (@ABS_prod A B r)) = r).
Proof.
  intros A B f.
  apply prop_ext.
  - intros [a [b ->]].
    rewrite -ABS_prod_mk_pair.
    reflexivity.
  - generalize (ABS_prod f).
    intros [a b] e.
    exists a.
    exists b.
    rewrite <- e.
    reflexivity.
Qed.

(****************************************************************************)
(* Alignment of the infinite type ind. *)
(****************************************************************************)

Definition ind : Type' := nat.

Definition ONE_ONE A B := @injective B A.

Lemma ONE_ONE_def {A B : Type'} : (@ONE_ONE A B) = (fun _2064 : A -> B => forall x1 : A, forall x2 : A, ((_2064 x1) = (_2064 x2)) -> x1 = x2).
Proof. exact erefl. Qed.

Require Import mathcomp.classical.functions. 
Definition ONTO {A B : Type'} (f : A -> B) := set_surj setT setT f.

Lemma ONTO_def {A B : Type'} : (@ONTO A B) = (fun _2069 : A -> B => forall y : B, exists x : A, y = (_2069 x)).
Proof.
  ext=>f H x ; last by case:(H x)=>x0 eq_fx0_x _ ; exists x0.
  by case: (H x Logic.I)=>x0 _ eq_fx0_x ; exists x0.
Qed.

Lemma axiom_6 : exists f : ind -> ind, (@ONE_ONE ind ind f) /\ (~ (@ONTO ind ind f)).
Proof.
  rewrite ONTO_def.
  exists S.
  split.
  - exact eq_add_S.
  - intro h.
    generalize (h 0).
    intros [x hx].
    discriminate.
Qed.

Definition IND_SUC_pred := fun f : ind -> ind => exists z : ind, (forall x1 : ind, forall x2 : ind, ((f x1) = (f x2)) = (x1 = x2)) /\ (forall x : ind, ~ ((f x) = z)).

Definition IND_SUC := ε IND_SUC_pred.

Lemma IND_SUC_def : IND_SUC = (@ε (ind -> ind) (fun f : ind -> ind => exists z : ind, (forall x1 : ind, forall x2 : ind, ((f x1) = (f x2)) = (x1 = x2)) /\ (forall x : ind, ~ ((f x) = z)))).
Proof. exact erefl. Qed.

Lemma IND_SUC_ex : exists f, IND_SUC_pred f.
Proof.
  destruct axiom_6 as [f [h1 h2]]. exists f.
  rewrite ONTO_def in h2. rewrite <- existsNE in h2. destruct h2 as [z h2]. exists z.
  split.
  intros x y. apply prop_ext. apply h1. intro e. rewrite e. reflexivity.
  rewrite <- forallNE in h2. intros x e. apply (h2 x). symmetry. exact e.
Qed.

Lemma IND_SUC_prop : IND_SUC_pred IND_SUC.
Proof. unfold IND_SUC. apply ε_spec. apply IND_SUC_ex. Qed.

Lemma IND_SUC_inj : ONE_ONE IND_SUC.
Proof.
  generalize IND_SUC_prop. intros [z [h1 h2]]. intros x y e. rewrite <- h1.
  exact e.
Qed.

Definition IND_0_pred := fun z : ind => (forall x1 : ind, forall x2 : ind, ((IND_SUC x1) = (IND_SUC x2)) = (x1 = x2)) /\ (forall x : ind, ~ ((IND_SUC x) = z)).

Definition IND_0 := ε IND_0_pred.

Lemma IND_0_def : IND_0 = (@ε ind (fun z : ind => (forall x1 : ind, forall x2 : ind, ((IND_SUC x1) = (IND_SUC x2)) = (x1 = x2)) /\ (forall x : ind, ~ ((IND_SUC x) = z)))).
Proof. exact erefl. Qed.

Lemma IND_0_ex : exists z, IND_0_pred z.
Proof.
  generalize IND_SUC_prop. intros [z [h1 h2]]. exists z. split. exact h1. exact h2.
Qed.

Lemma IND_0_prop : IND_0_pred IND_0.
Proof. unfold IND_0. apply ε_spec. apply IND_0_ex. Qed.

Lemma IND_SUC_neq_0 i : IND_SUC i <> IND_0.
Proof. generalize IND_0_prop. intros [h1 h2]. apply h2. Qed.

(****************************************************************************)
(* NUM_REP. *)
(****************************************************************************)

Inductive NUM_REP : ind -> Prop :=
  | NUM_REP_0 : NUM_REP IND_0
  | NUM_REP_S i : NUM_REP i -> NUM_REP (IND_SUC i).

Lemma NUM_REP_def : NUM_REP = (fun a : ind => forall NUM_REP' : ind -> Prop, (forall a' : ind, ((a' = IND_0) \/ (exists i : ind, (a' = (IND_SUC i)) /\ (NUM_REP' i))) -> NUM_REP' a') -> NUM_REP' a).
Proof. ind_align. Qed.

(****************************************************************************)
(* Alignment of the type of natural numbers. *)
(****************************************************************************)

From Stdlib Require Import NArith.BinNat micromega.Lia.
Open Scope N_scope.

(* Not using is_Type' to keep its boolean equality *)
HB.instance Definition _ := gen_choiceMixin N.

HB.instance Definition _ := isPointed.Build _ 0.

Lemma N0_or_succ n : n = 0 \/ exists p, n = N.succ p.
Proof. case:(pselect (n=0))=>H. auto. right. exists (N.pred n). lia. Qed.

Lemma recursion_succ A (a:A) f n :
  N.recursion a f (N.succ n) = f n (N.recursion a f n).
Proof.
  apply N.recursion_succ. reflexivity.
  intros n1 n2 n12 a1 a2 a12. subst n2. subst a2. reflexivity.
Qed.

Definition dest_num := N.recursion IND_0 (fun _ r => IND_SUC r).

Lemma dest_num0 : dest_num 0 = IND_0.
Proof. unfold dest_num. apply N.recursion_0. Qed.

Lemma dest_numS n : dest_num (N.succ n) = IND_SUC (dest_num n).
Proof. unfold dest_num at 1. apply recursion_succ. Qed.

Lemma dest_num_inj : injective dest_num.
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
  ext=> n ; first elim => [|_ _ [m ->]] ; first by exists 0.
  - by exists (N.succ m) ; rewrite dest_numS.
  - move=> [m ->] ; move: m ; apply N.peano_ind.
    + rewrite dest_num0 ; exact NUM_REP_0.
    + move=>m Hm ; rewrite dest_numS ; exact (NUM_REP_S Hm).
Qed.

Lemma NUM_REP_dest_num k : NUM_REP (dest_num k).
Proof. rewrite NUM_REP_eq_dest_num_img. exists k. reflexivity. Qed.

Definition mk_num_pred i n := i = dest_num n.

Definition mk_num i := ε (mk_num_pred i).

Lemma mk_num_ex i : NUM_REP i -> exists n, mk_num_pred i n.
Proof.
  induction 1.
  exists 0. reflexivity.
  destruct IHNUM_REP as [n hn]. exists (N.succ n). unfold mk_num_pred.
  rewrite hn dest_numS. reflexivity.
Qed.

Lemma mk_num_prop i : NUM_REP i -> dest_num (mk_num i) = i.
Proof. intro hi. symmetry. apply (ε_spec (mk_num_ex hi)). Qed.

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

Lemma _0_def : 0 = (mk_num IND_0).
Proof. symmetry. exact (axiom_7 0). Qed.

Lemma mk_num_S : forall i, NUM_REP i -> mk_num (IND_SUC i) = N.succ (mk_num i).
Proof.
  intros i hi. rewrite NUM_REP_eq_dest_num_img in hi. destruct hi as [n hn].
  subst i. rewrite mk_num_dest_num -dest_numS mk_num_dest_num. reflexivity.
Qed.

Lemma SUC_def : N.succ = (fun _2104 : N => mk_num (IND_SUC (dest_num _2104))).
Proof.
  symmetry. ext=>x. rewrite mk_num_S. 2: apply NUM_REP_dest_num.
  apply f_equal. apply axiom_7.
Qed.

(****************************************************************************)
(* tactics to align recursive functions on N. *)
(****************************************************************************)

(* N_rec_alignk for k between 1 and 3 replaces a goal of the form 
     f = ε P uv with two goals PO f and PS f whenever P = PO /\ PS
   totally defines the function by peano recursion on the kth argument. *)
(* These tactics only work for functions with 3 or less arguments. *)

(* Funnily enough, we must first redefine align_ε *)

Tactic Notation "align_ε" :=
  let rec aux :=
    lazymatch goal with
    | |- _ ?x = ε _ ?x => apply (f_equal (fun f => f x)) ; aux
    | |- ?a = ε _ ?r =>
        (* Replace the goal by (fun _ => a = ε ?P) *)
        apply (f_equal (fun g => g r) (x := fun _ => a)) ;
        aux ;
        [ let uv := fresh in
          intro uv ; clear uv 

        | let a' := fresh in
          let uv := fresh in
          let H' := fresh in
          let H := fresh in
          intros a' H H' ; ext 1=> uv ;
          specialize (H uv) ; (* As [P] starts with [forall uv] *)
          specialize (H' uv) ;
          simpl ((fun _ => a) uv) in * ; (* Simplifies to [a] so that [uv] only appears in [a' uv] *)
          gobble a' uv ;
          revert a' H H' (* Revert [a'], [P a] and [P a'] to reuse them in other tactics *)
        ]
    | |- ?a = ε ?P => apply align_ε (* Replaces the goal [a = ε P] with two goals [P a] and
                                       [forall x, P a => P x => x = a]. *)
    end
  in aux.

Ltac N_rec_align1 :=
  align_ε ; (* At this state, we have two goals : P f and P f -> P f' -> f = f'.
                We now assume that P is of the form
                g 0 = x /\ forall n, g (Succ n) = y for some x and y. *)
  [ split ; auto (* since it is a conjunction, we can split *)
  | let f' := fresh in
    let n := fresh in
    let HO := fresh in
    let HS := fresh in
    let HO' := fresh in
    let HS' := fresh in
    let IHn := fresh in
    intros f' (HO , HS) (HO' , HS') ; (* Naming specifically each clause in H and H'. *)
    ext 1 => n ; match goal with |- ?f n = f' n => 
      revert n ; apply (N.peano_rec (fun n => (f n = f' n))) ; try intros n IHn ;
      do 2 try apply funext=>? ; [
        rewrite HO ; rewrite HO' (* f 0 and f' 0 are replaced with the same value. This ensures that we are inducting on the correct variable otherwise rewriting would fail. *)
      | rewrite HS ; rewrite HS' ; try rewrite <- IHn (* Same as above. *)
      ] ; auto end
        ] .
  (* If all works correctly we have two goals left, PO f and PS f.
     PO f is often already solved, and in easy cases, so is PS f. *) 

(* N_rec_align2 and N_rec_align3 are very similar. *)

Ltac N_rec_align2 :=
  align_ε ; [ split ; auto
  | let f' := fresh in
    let n := fresh in
    let a := fresh in
    let HO := fresh in
    let HS := fresh in
    let HO' := fresh in
    let HS' := fresh in
    let IHn := fresh in
    intros f' (HO , HS) (HO' , HS') ; ext 2=> a n ;
    match goal with |- ?f a n = f' a n =>
      revert n a ; apply (N.peano_rec (fun n => forall a, f a n = f' a n)) ; [
        intro a ; try apply funext=>? ; rewrite HO ; rewrite HO' 
      | intros n IHn a ; try apply funext=>? ;
        rewrite HS ; rewrite HS' ; try rewrite <- IHn ] ; auto end
        ] .

Ltac N_rec_align3 :=
  align_ε ; [ split ; auto
  | let f' := fresh in
    let n := fresh in
    let a := fresh in
    let b := fresh in
    let HO := fresh in
    let HS := fresh in
    let HO' := fresh in
    let HS' := fresh in
    let IHn := fresh in
    intros f' (HO , HS) (HO' , HS') ; ext => a b n ;
    match goal with |- ?f a b n = f' a b n =>
      revert n a b ; apply (N.peano_rec (fun n => forall a b, f a b n = f' a b n)) ; [
        intros a b ; rewrite HO ; rewrite HO'
      | intros n IHn a b ; rewrite HS ; rewrite HS' ; try rewrite <- IHn ] ; auto end
        ].

Ltac N_rec_align :=
  try N_rec_align1 ;
  try N_rec_align2 ;
  try N_rec_align3.

(****************************************************************************)
(* Alignment of mathematical functions on natural numbers with N. *)
(****************************************************************************)

Definition NUMERAL (x : N) := x.

Lemma NUMERAL_def : NUMERAL = (fun _2128 : N => _2128).
Proof. exact erefl. Qed.

Definition BIT0 := N.double.

Lemma BIT0_def : BIT0 = @ε (arr N N) (fun y0 : N -> N => ((y0 (NUMERAL N0)) = (NUMERAL N0)) /\ (forall y1 : N, (y0 (N.succ y1)) = (N.succ (N.succ (y0 y1))))).
Proof.
  unfold BIT0 , NUMERAL.
  N_rec_align. lia.
Qed.

Definition BIT1 := fun n : N => N.succ (BIT0 n).

Ltac numfold := unfold NUMERAL, BIT1, BIT0 in *.
Ltac numsimp := numfold ; simpl.

(* automatically unfold them before lia. *)
Ltac lia := numfold ; Stdlib.micromega.Lia.lia.

Lemma BIT1_def : BIT1 = (fun _2143 : N => N.succ (BIT0 _2143)).
Proof. exact erefl. Qed.

Lemma BIT1_eq_succ_double : BIT1 = N.succ_double. 
Proof. ext ; lia. Qed.

Lemma PRE_def : N.pred = (@ε (arr (prod N (prod N N)) (arr N N)) (fun PRE' : (prod N (prod N N)) -> N -> N => forall _2151 : prod N (prod N N), ((PRE' _2151 (NUMERAL N0)) = (NUMERAL N0)) /\ (forall n : N, (PRE' _2151 (N.succ n)) = n)) (@pair N (prod N N) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))))).
Proof.
  numsimp. N_rec_align. lia.
Qed.

Lemma add_def : N.add = (@ε (arr N (arr N (arr N N))) (fun add' : N -> N -> N -> N => forall _2155 : N, (forall n : N, (add' _2155 (NUMERAL N0) n) = n) /\ (forall m : N, forall n : N, (add' _2155 (N.succ m) n) = (N.succ (add' _2155 m n)))) (NUMERAL (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))).
Proof.
  N_rec_align. lia.
Qed.

Lemma mul_def : N.mul = (@ε (arr N (arr N (arr N N))) (fun mul' : N -> N -> N -> N => forall _2186 : N, (forall n : N, (mul' _2186 (NUMERAL N0) n) = (NUMERAL N0)) /\ (forall m : N, forall n : N, (mul' _2186 (N.succ m) n) = (N.add (mul' _2186 m n) n))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))).
Proof.
  numsimp. N_rec_align. lia. 
Qed.

Lemma EXP_def : N.pow = (@ε (arr (prod N (prod N N)) (arr N (arr N N))) (fun EXP' : (prod N (prod N N)) -> N -> N -> N => forall _2224 : prod N (prod N N), (forall m : N, EXP' _2224 m (NUMERAL N0) = NUMERAL (BIT1 N0)) /\ (forall m : N, forall n : N, (EXP' _2224 m (N.succ n)) = (N.mul m (EXP' _2224 m n)))) (@pair N (prod N N) (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0))))))) (@pair N N (BIT0 (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 0))))))) (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))))).
Proof.
  N_rec_align. exact N.pow_succ_r'.
Qed.

Lemma le_def : N.le = (@ε (arr (prod N N) (arr N (arr N Prop))) (fun le' : (prod N N) -> N -> N -> Prop => forall _2241 : prod N N, (forall m : N, (le' _2241 m (NUMERAL N0)) = (m = (NUMERAL N0))) /\ (forall m : N, forall n : N, (le' _2241 m (N.succ n)) = ((m = (N.succ n)) \/ (le' _2241 m n)))) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 (BIT1 0))))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT1 (BIT1 0))))))))).
Proof.
  N_rec_align.
  - intro n.
    apply propext.
    exact (N.le_0_r n).
  - intros n m.
    apply propext.
    rewrite or_comm.
    exact (N.le_succ_r n m).
Qed.

Lemma lt_def : N.lt = (@ε (arr N (arr N (arr N Prop))) (fun lt : N -> N -> N -> Prop => forall _2248 : N, (forall m : N, (lt _2248 m (NUMERAL N0)) = False) /\ (forall m : N, forall n : N, (lt _2248 m (N.succ n)) = ((m = n) \/ (lt _2248 m n)))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 (BIT1 0)))))))).
Proof.
  N_rec_align.
  - intro n.
    rewrite is_False.
    exact (N.nlt_0_r n).
  - intros n m.
    apply propext.
    rewrite N.lt_succ_r.
    rewrite or_comm.
    exact (N.lt_eq_cases n m).
Qed.

Lemma ge_def : N.ge = (fun _2249 : N => fun _2250 : N => N.le _2250 _2249).
Proof. ext;lia. Qed.

Lemma gt_def : N.gt = (fun _2261 : N => fun _2262 : N => N.lt _2262 _2261).
Proof. ext;lia. Qed.

Lemma N0_le_eq_True y : N.le 0 y = True.
Proof. rewrite is_True. exact (N.le_0_l y). Qed.

Lemma succ_le_0_is_False x : N.le (N.succ x) 0 = False.
Proof. rewrite is_False. exact (N.nle_succ_0 x). Qed.

Lemma succ_eq_0_is_False x : (N.succ x = N0) = False.
Proof. rewrite is_False. exact (N.neq_succ_0 x).  Qed.

Lemma le_succ_succ x y : N.le (N.succ x) (N.succ y) = N.le x y.
Proof. symmetry. apply propext. exact (N.succ_le_mono x y). Qed.

Lemma MAX_def : N.max = (fun _2273 : N => fun _2274 : N => @COND N (N.le _2273 _2274) _2274 _2273).
Proof.
  ext=>x y. if_intro ; lia.
Qed.

Lemma MIN_def : N.min = (fun _2285 : N => fun _2286 : N => @COND N (N.le _2285 _2286) _2285 _2286).
Proof.
  ext=>x y. if_intro ; lia.
Qed.

Lemma minus_def : N.sub = (@ε (arr N (arr N (arr N N))) (fun pair' : N -> N -> N -> N => forall _2766 : N, (forall m : N, (pair' _2766 m (NUMERAL N0)) = m) /\ (forall m : N, forall n : N, (pair' _2766 m (N.succ n)) = (N.pred (pair' _2766 m n)))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 0)))))))).
Proof.
  N_rec_align.
  - exact N.sub_0_r.
  - exact N.sub_succ_r.
Qed.

Definition fact := N.peano_rect (fun _ => N) 1 (fun n r => N.succ n * r).

Lemma FACT_def : fact = @ε ((prod N (prod N (prod N N))) -> N -> N) (fun FACT' : (prod N (prod N (prod N N))) -> N -> N => forall _2944 : prod N (prod N (prod N N)), ((FACT' _2944 (NUMERAL 0%N)) = (NUMERAL (BIT1 0%N))) /\ (forall n : N, (FACT' _2944 (N.succ n)) = (N.mul (N.succ n) (FACT' _2944 n)))) (@pair N (prod N (prod N N)) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0%N)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0%N)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0%N)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0%N))))))))))).
Proof.
  numsimp. N_rec_align. 
  intro n. unfold fact. now rewrite N.peano_rect_succ.
Qed.

Lemma Nadd_sub a b : a + b - a = b. Proof. lia. Qed.

Lemma Nswap_add_sub a a' b : a' <= a -> a + b - a' = a - a' + b. Proof. lia. Qed.

Lemma Ndivmod_unicity k k' r r' q :
  r < q -> r' < q -> k * q + r = k' * q + r' -> k = k' /\ r = r'.
Proof.
  intros h h' e.
  assert (e2 : k * q + r - k' * q = k' * q + r' - k' * q). lia.
  case:(pselect (N.lt k k'))=>H. nia.
  rewrite -> Nadd_sub, Nswap_add_sub, <- N.mul_sub_distr_r in e2. nia. nia.
Qed.
Arguments Ndivmod_unicity [_] [_] [_] [_] _.

Lemma DIV_def : N.div = (@ε (arr (prod N (prod N N)) (arr N (arr N N))) (fun q : (prod N (prod N N)) -> N -> N -> N => forall _3086 : prod N (prod N N), exists r : N -> N -> N, forall m : N, forall n : N, @COND Prop (n = (NUMERAL N0)) (((q _3086 m n) = (NUMERAL N0)) /\ ((r m n) = m)) ((m = (N.add (N.mul (q _3086 m n) n) (r m n))) /\ (N.lt (r m n) n))) (@pair N (prod N N) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))).
Proof.
  cbn.
  align_ε.
  - exists N.modulo.
    intros m n.
    if_intro=>h.
    + rewrite h.
      split.
      * exact (N.div_0_r m).
      * exact (N.mod_0_r m).
    + split.
      * rewrite N.mul_comm.
        exact (N.Div0.div_mod m n).
      * exact (N.mod_lt m n h).
  - intros div' _ h.
    destruct h as [mod' h].
    ext=>x y.
    apply (if_elim (h x y)); intros c [d m].
    + rewrite d c.
      exact (N.div_0_r x).
    + apply (@proj1 _ (x mod y = mod' x y)).
      apply (Ndivmod_unicity y).
      * exact (N.mod_lt x y c).
      * exact m.
      * rewrite <- d, N.mul_comm.
        exact (esym (N.Div0.div_mod x y)).
Qed.

Lemma MOD_def : N.modulo = (@ε (arr (prod N (prod N N)) (arr N (arr N N))) (fun r : (prod N (prod N N)) -> N -> N -> N => forall _3087 : prod N (prod N N), forall m : N, forall n : N, @COND Prop (n = (NUMERAL 0)) (((N.div m n) = (NUMERAL 0)) /\ ((r _3087 m n) = m)) ((m = (N.add (N.mul (N.div m n) n) (r _3087 m n))) /\ (N.lt (r _3087 m n) n))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))))).
Proof.
  cbn.
  align_ε.
  - intros m n. if_intro=>h.
    + rewrite h.
      split.
      * exact (N.div_0_r m).
      * exact (N.mod_0_r m).
    + split.
      * rewrite N.mul_comm.
        exact (N.Div0.div_mod m n).
      * exact (N.mod_lt m n h).
  - intros mod' _ h.
    ext=> x y.
    apply (if_elim (h x y)); intros c [d m].
    + rewrite m c.
      exact (N.mod_0_r x).
    + apply (@proj2 (x / y = x / y)).
      apply (Ndivmod_unicity y).
      * exact (N.mod_lt x y c).
      * exact m.
      * rewrite <- d, N.mul_comm.
        exact (esym (N.Div0.div_mod x y)).
Qed.

(****************************************************************************)
(* Alignment of the Even and Odd predicates. *)
(****************************************************************************)

Lemma NEven0: N.Even 0 = True.
Proof.
  rewrite is_True.
  apply (proj1 (N.even_spec 0)).
  reflexivity.
Qed.

Lemma NOdd0: N.Odd 0 = False.
Proof.
  rewrite is_False.
  apply (proj1 (not_iff_compat (N.odd_spec 0))).
  rewrite (Bool.not_true_iff_false _).
  exact N.odd_0.
Qed.

Lemma NEvenS: forall n: N, N.Even (N.succ n) = ~ N.Even n.
Proof.
  intro n.
  rewrite (propext (N.Even_succ n)).
  apply prop_ext.
    - unfold N.Odd.
      apply ex_ind.
      intros m o.
      rewrite o -(propext (N.even_spec _)) (Bool.not_true_iff_false _).
      exact (N.even_odd _).
    - case (N.Even_or_Odd n).
      + exact (absurd _).
      + trivial.
Qed.

Lemma NOddS: forall n: N, N.Odd (N.succ n) = ~ N.Odd n.
Proof.
  intro n.
  rewrite (propext (N.Odd_succ n)).
  apply prop_ext.
  - unfold N.Even.
    apply ex_ind.
    intros m o.
    rewrite o -(propext (N.odd_spec _)) (Bool.not_true_iff_false _).
    exact (N.odd_even _).
  - case (N.Even_or_Odd n).
    + trivial.
    + exact (absurd _).
Qed.

Lemma EVEN_def : N.Even = @ε ((prod N (prod N (prod N N))) -> N -> Prop) (fun EVEN' : (prod N (prod N (prod N N))) -> N -> Prop => forall _2603 : prod N (prod N (prod N N)), ((EVEN' _2603 (NUMERAL 0%N)) = True) /\ (forall n : N, (EVEN' _2603 (N.succ n)) = (~ (EVEN' _2603 n)))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0%N)))))))) (@pair N (prod N N) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0%N)))))))) (@pair N N (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0%N)))))))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0%N))))))))))).
Proof.
  numsimp. N_rec_align.
  exact (NEven0). exact (NEvenS).
Qed.

Lemma ODD_def: N.Odd = @ε ((prod N (prod N N)) -> N -> Prop) (fun ODD' : (prod N (prod N N)) -> N -> Prop => forall _2607 : prod N (prod N N), ((ODD' _2607 (NUMERAL 0%N)) = False) /\ (forall n : N, (ODD' _2607 (N.succ n)) = (~ (ODD' _2607 n)))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0%N)))))))) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0%N)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0%N)))))))))).
Proof.
  numsimp. N_rec_align.
  exact (NOdd0). exact (NOddS).
Qed.

(****************************************************************************)
(* NUMPAIR(x,y) = 2^x(2y+1): bijection between N² and N-{0}. *)
(****************************************************************************)

Definition NUMPAIR := fun x : N => fun y : N => N.mul (N.pow (NUMERAL (BIT0 (BIT1 0))) x) (N.add (N.mul (NUMERAL (BIT0 (BIT1 0))) y) (NUMERAL (BIT1 0))).

Lemma NUMPAIR_def : NUMPAIR = (fun _17487 : N => fun _17488 : N => N.mul (N.pow (NUMERAL (BIT0 (BIT1 N0))) _17487) (N.add (N.mul (NUMERAL (BIT0 (BIT1 N0))) _17488) (NUMERAL (BIT1 N0)))).
Proof. exact erefl. Qed.

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
  unfold NUMPAIR. numfold. rewrite double_0 succ_0 double_1.
  intro e.

  case:(pselect (x1 < x2))=>h. rewrite lt2le in h.
  apply False_rec. destruct (le_is_add h) as [k hk]. subst x2.
  generalize (f_equal (fun x => N.div x (2 ^ x1)) e).
  rewrite -> NDIV_MULT, N.pow_add_r, N.pow_succ_r, (N.mul_comm 2 (2 ^ x1)),
    <- !N.mul_assoc, NDIV_MULT.
  lia. lia. lia. lia.

  case:(pselect (x2 < x1))=>i. rewrite lt2le in i.
  apply False_rec. destruct (le_is_add i) as [k hk]. subst x1.
  generalize (f_equal (fun x => N.div x (2 ^ x2)) e).
  rewrite ->NDIV_MULT, N.pow_add_r, N.pow_succ_r, (N.mul_comm 2 (2 ^ x2)),
    <- !N.mul_assoc, NDIV_MULT.
  lia. lia. lia. lia.

  assert (j: x1 = x2). lia. subst x2. split. reflexivity. nia.
Qed.

Lemma NUMPAIR_nonzero x y : NUMPAIR x y <> 0.
Proof.
  unfold NUMPAIR. lia.
Qed.

(****************************************************************************)
(* Inverse of NUMPAIR. *)
(****************************************************************************)

(* used for some embeddings in the definition of recspace *)

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
  destruct (INJ_INVERSE2 NUMPAIR_INJ) as [fst [snd h]].
  assert (i : exists q, NUMFST0_pred q). exists fst. exists snd. assumption.
  generalize (ε_spec i). fold NUMFST0. unfold NUMFST0_pred.
  intros [snd' h']. destruct (h' x y) as [j k]. assumption.
Qed.

Definition NUMSND0_pred :=  fun Y : N -> N => exists X : N -> N, forall x : N, forall y : N, ((X (NUMPAIR x y)) = x) /\ ((Y (NUMPAIR x y)) = y).

Definition NUMSND0 := ε NUMSND0_pred.

Lemma NUMSND0_NUMPAIR x y : NUMSND0 (NUMPAIR x y) = y.
Proof.
  destruct (INJ_INVERSE2 NUMPAIR_INJ) as [fst [snd h]].
  assert (i : exists x, NUMSND0_pred x). exists snd. exists fst. assumption.
  generalize (ε_spec i). fold NUMSND0. unfold NUMSND0_pred.
  intros [fst' h']. destruct (h' x y) as [j k]. assumption.
Qed.

Definition NUMFST := @ε ((prod N (prod N (prod N (prod N (prod N N))))) -> N -> N) (fun X : (prod N (prod N (prod N (prod N (prod N N))))) -> N -> N => forall _17340 : prod N (prod N (prod N (prod N (prod N N)))), exists Y : N -> N, forall x : N, forall y : N, ((X _17340 (NUMPAIR x y)) = x) /\ ((Y (NUMPAIR x y)) = y)) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))))).

Lemma NUMFST_def : NUMFST = (@ε ((prod N (prod N (prod N (prod N (prod N N))))) -> N -> N) (fun X : (prod N (prod N (prod N (prod N (prod N N))))) -> N -> N => forall _17503 : prod N (prod N (prod N (prod N (prod N N)))), exists Y : N -> N, forall x : N, forall y : N, ((X _17503 (NUMPAIR x y)) = x) /\ ((Y (NUMPAIR x y)) = y)) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))))))))).
Proof. exact erefl. Qed.

Lemma NUMFST_NUMPAIR x y : NUMFST (NUMPAIR x y) = x.
Proof.
  unfold NUMFST. numsimp. generalize (78, (85, (77, (70, (83, 84))))).
  generalize (prod N (prod N (prod N (prod N (prod N N))))); intros A a.
  match goal with |- ε ?x _ _ = _ => set (Q := x); set (fst := ε Q) end.
  assert (i: exists x, Q x). exists (fun _ => NUMFST0). unfold Q. intros _. exists NUMSND0.
  intros x' y'. rewrite NUMFST0_NUMPAIR NUMSND0_NUMPAIR. auto.
  generalize (ε_spec i). change (Q fst -> fst a (NUMPAIR x y) = x). intro h.
  destruct (h a) as [snd j]. destruct (j x y) as [j1 j2]. exact j1.
Qed.

Definition NUMSND1_pred :=  fun Y : N -> N => forall x : N, forall y : N, ((NUMFST (NUMPAIR x y)) = x) /\ ((Y (NUMPAIR x y)) = y).

Definition NUMSND1 := ε NUMSND1_pred.

Lemma NUMSND1_NUMPAIR x y : NUMSND1 (NUMPAIR x y) = y.
Proof.
  destruct (INJ_INVERSE2 NUMPAIR_INJ) as [fst [snd h]].
  assert (i : exists x, NUMSND1_pred x). exists snd. unfold NUMSND1_pred.
  intros x' y'. rewrite NUMFST_NUMPAIR. destruct (h x' y') as [h1 h2]. auto.
  generalize (ε_spec i). fold NUMSND1. unfold NUMSND1_pred. intro j.
  destruct (j x y) as [j1 j2]. exact j2.
Qed.

Definition NUMSND := @ε ((prod N (prod N (prod N (prod N (prod N N))))) -> N -> N) (fun Y : (prod N (prod N (prod N (prod N (prod N N))))) -> N -> N => forall _17341 : prod N (prod N (prod N (prod N (prod N N)))), forall x : N, forall y : N, ((NUMFST (NUMPAIR x y)) = x) /\ ((Y _17341 (NUMPAIR x y)) = y)) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))))))).

Lemma NUMSND_def : NUMSND = (@ε ((prod N (prod N (prod N (prod N (prod N N))))) -> N -> N) (fun Y : (prod N (prod N (prod N (prod N (prod N N))))) -> N -> N => forall _17504 : prod N (prod N (prod N (prod N (prod N N)))), forall x : N, forall y : N, ((NUMFST (NUMPAIR x y)) = x) /\ ((Y _17504 (NUMPAIR x y)) = y)) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))))))))).
Proof. exact erefl. Qed.

Lemma NUMSND_NUMPAIR x y : NUMSND (NUMPAIR x y) = y.
Proof.
  unfold NUMSND. numsimp. generalize (78, (85, (77, (83, (78, 68))))).
  generalize (prod N (prod N (prod N (prod N (prod N N))))); intros A a.
  match goal with |- ε ?x _ _ = _ => set (Q := x); set (snd := ε Q) end.
  assert (i: exists x, Q x). exists (fun _ => NUMSND1). unfold Q. intros _.
  intros x' y'. rewrite NUMFST_NUMPAIR NUMSND1_NUMPAIR. auto.
  generalize (ε_spec i). change (Q snd -> snd a (NUMPAIR x y) = y). intro h.
  destruct (h a x y) as [h1 h2]. exact h2.
Qed.

(****************************************************************************)
(* NUMSUM(b,n) = if b then 2n+1 else 2n : bijection between BxN and N. *)
(****************************************************************************)

(* used for some embeddings in the definition of recspace *)

Definition NUMSUM := fun b : Prop => fun n : N => @COND N b (N.succ (N.mul (NUMERAL (BIT0 (BIT1 0))) n)) (N.mul (NUMERAL (BIT0 (BIT1 0))) n).

Lemma NUMSUM_def : NUMSUM = (fun _17505 : Prop => fun _17506 : N => @COND N _17505 (N.succ (N.mul (NUMERAL (BIT0 (BIT1 N0))) _17506)) (N.mul (NUMERAL (BIT0 (BIT1 N0))) _17506)).
Proof. exact erefl. Qed.

Definition NUMLEFT n := if N.even n then False else True.

Definition NUMRIGHT n := if N.even n then n / 2 else (n - 1) / 2.

Lemma even_double n : N.even (2 * n) = true.
Proof. rewrite N.even_spec. exists n. reflexivity. Qed.

Lemma NUMLEFT_NUMSUM b n : NUMLEFT (NUMSUM b n) = b.
Proof.
  unfold NUMSUM, NUMLEFT. numfold.
  if_intro=>H ; rewrite double_0 succ_0 double_1 ;
  apply prop_ext ; intro H' ; try easy.
  - rewrite N.even_succ N.odd_mul N.odd_2. reflexivity.
  - rewrite even_double in H'. destruct H'.
Qed.

Lemma succ_minus_1 x : N.succ x - 1 = x.
Proof. lia. Qed.

Lemma NUMRIGHT_NUMSUM b n : NUMRIGHT (NUMSUM b n) = n.
Proof.
  unfold NUMSUM, NUMRIGHT. numfold.
  if_intro=>H ; rewrite double_0 succ_0 double_1.
  - rewrite N.even_succ N.odd_mul N.odd_2 succ_minus_1 NDIV_MULT.
    reflexivity. discriminate.
  - rewrite even_double NDIV_MULT. reflexivity. discriminate.
Qed.

Lemma Nplus_1_minus_1 x : x + 1 - 1 = x.
Proof. lia. Qed.

Lemma NUMSUM_surjective n : exists b x, n = NUMSUM b x.
Proof.
  exists (NUMLEFT n). exists (NUMRIGHT n). unfold NUMSUM, NUMLEFT, NUMRIGHT. numfold.
  case_eq (N.even n); intro h ; if_triv.
  - rewrite N.even_spec in h. destruct h as [k h]. subst n.
    rewrite NDIV_MULT. reflexivity. lia.
  - apply eq_false_negb_true in h. change (N.odd n = true) in h.
    rewrite N.odd_spec in h. destruct h as [k h]. subst. rewrite Nplus_1_minus_1.
    rewrite NDIV_MULT. lia. lia.
Qed.

Lemma NUMLEFT_def : NUMLEFT = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> N -> Prop) (fun X : (prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> N -> Prop => forall _17372 : prod N (prod N (prod N (prod N (prod N (prod N N))))), exists Y : N -> N, forall x : Prop, forall y : N, ((X _17372 (NUMSUM x y)) = x) /\ ((Y (NUMSUM x y)) = y)) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))))))).
Proof.
  cbn.
  align_ε.
  - exists NUMRIGHT.
    intros x y.
    split.
    + exact (NUMLEFT_NUMSUM x y).
    + exact (NUMRIGHT_NUMSUM x y).
  - intros NUMLEFT' _ h.
    destruct h as [NUMRIGHT' h].
    ext 1=>s.
    destruct (NUMSUM_surjective s) as [b [x k]].
    rewrite k (NUMLEFT_NUMSUM b x) (proj1 (h b x)).
    reflexivity.
Qed.

Lemma NUMRIGHT_def : NUMRIGHT = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> N -> N) (fun Y : (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> N -> N => forall _17373 : prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))), forall x : Prop, forall y : N, ((NUMLEFT (NUMSUM x y)) = x) /\ ((Y _17373 (NUMSUM x y)) = y)) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N N)))))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))))))))).
Proof.
  cbn.
  align_ε.
  - split.
    + exact (NUMLEFT_NUMSUM x y).
    + exact (NUMRIGHT_NUMSUM x y).
  - intros NUMRIGHT' _ h.
    ext =>s.
    destruct (NUMSUM_surjective s) as [b [x k]].
    rewrite k (NUMRIGHT_NUMSUM b x) (proj2 (h b x)).
    reflexivity.
Qed.

(****************************************************************************)
(* Alignment of well-foundedness.
HOL Light: non-empty subsets has minimal, Rocq: has induction *)
(****************************************************************************)

Require Import Corelib.Init.Wf.

Definition well_founded := Corelib.Init.Wf.well_founded.

Lemma WF_def {A : Type'} : (@well_founded A) = (fun _6923 : A -> A -> Prop => forall P : A -> Prop, (exists x : A, P x) -> exists x : A, (P x) /\ (forall y : A, (_6923 y x) -> ~ (P y))).
Proof.
  ext => R H.
  - intros X ne.
    destruct ne as [x Hx].
    rewrite <- (notE _); intro goal.
    case (prop_degen (forall y: A, ~ X y)); intro h.
    + rewrite is_True in h.
      assert (~ X x) by exact (h x).
      contradiction.
    + rewrite is_False in h.
      apply h.
      apply (well_founded_induction H).
      intros y g Xy.
      apply goal.
      exists y.
      split; assumption.
  - unfold well_founded.
    case (prop_degen (forall a : A, Acc R a)); intro h.
    + rewrite is_True in h.
      assumption.
    + rewrite -> is_False, <- existsNE in h.
      apply except.
      assert (G: exists x : A, ~ Acc R x /\ (forall y : A, R y x -> ~ ~ Acc R y))
        by apply (H _ h).
      destruct G as [x Gx].
      destruct Gx as [H0 H1].
      apply H0.
      apply Acc_intro.
      intros y Ryx.
      rewrite <- (notE _).
      apply H1.
      assumption.
Qed.

(****************************************************************************)
(* Alignment of  measures, that is functions A -> N which creates a wf order by inverse image *)
(****************************************************************************)

Require Import Stdlib.Arith.PeanoNat.

Lemma inj_lt m n: (N.to_nat m > N.to_nat n)%coq_nat = (n < m).
Proof.
  ext => h.
  - unfold N.lt.
    rewrite (Nnat.N2Nat.inj_compare n m).
    apply (proj2 (Nat.compare_lt_iff _ _)).
    assumption.
  - apply (proj1 (Nat.compare_lt_iff _ _)).
    rewrite <- (Nnat.N2Nat.inj_compare n m).
    unfold N.lt in h.
    assumption.
Qed.

(*Definition MEASURE {A : Type'} : (A -> N) -> A -> A -> Prop := fun f : A -> N => @Wf_nat.gtof A (fun x : A => N.to_nat (f x)).

Lemma MEASURE_def {A : Type'} : (fun f : A -> N => @Wf_nat.gtof A (fun x : A => N.to_nat (f x))) = (fun _8094 : A -> N => fun x : A => fun y : A => N.lt (_8094 x) (_8094 y)).
Proof.
  apply fun_ext; intro f.
  unfold Wf_nat.gtof, MEASURE.
  ext x y.
  exact (inj_lt _ _).
Qed.*)

(****************************************************************************)
(* Alignment of recspace, the HOL-Light type used to encode inductive types. *)
(****************************************************************************)

(* recspace is basically an inductive type with one constant constructor BOTTOM
   and one recursive constructor CONSTR : N -> A -> (N -> recspace A) -> recspace A,
   defined through an embedding inside M := N -> A -> Prop.
   INJN, INJA and INJF embed N, A and N -> M inside M,
   which, together with INJP embedding M*M inside M, allows to embed 
   N * A * (N -> M). *)

Definition INJN {A : Type'} := fun x : N => fun n : N => fun a : A => n = x.

Lemma INJN_def {A : Type'} : (@INJN A) = (fun _17537 : N => fun n : N => fun a : A => n = _17537).
Proof. exact erefl. Qed.

Definition INJA {A : Type'} := fun x : A => fun n : N => fun b : A => b = x.

Lemma INJA_def {A : Type'} : (@INJA A) = (fun _17542 : A => fun n : N => fun b : A => b = _17542).
Proof. exact erefl. Qed.

Definition INJF {A : Type'} := fun f : N -> N -> A -> Prop => fun n : N => f (NUMFST n) (NUMSND n).

Lemma INJF_def {A : Type'} : (@INJF A) = (fun _17549 : N -> N -> A -> Prop => fun n : N => _17549 (NUMFST n) (NUMSND n)).
Proof. exact erefl. Qed.

Definition INJP {A : Type'} := fun f : N -> A -> Prop => fun g : N -> A -> Prop => fun n : N => fun a : A => @COND Prop (NUMLEFT n) (f (NUMRIGHT n) a) (g (NUMRIGHT n) a).

Lemma INJP_def {A : Type'} : (@INJP A) = (fun _17554 : N -> A -> Prop => fun _17555 : N -> A -> Prop => fun n : N => fun a : A => @COND Prop (NUMLEFT n) (_17554 (NUMRIGHT n) a) (_17555 (NUMRIGHT n) a)).
Proof. exact erefl. Qed.

Definition ZCONSTR {A : Type'} := fun n : N => fun a : A => fun f : N -> N -> A -> Prop => @INJP A (@INJN A (N.succ n)) (@INJP A (@INJA A a) (@INJF A f)).

Lemma ZCONSTR_def {A : Type'} : (@ZCONSTR A) = (fun _17566 : N => fun _17567 : A => fun _17568 : N -> N -> A -> Prop => @INJP A (@INJN A (N.succ _17566)) (@INJP A (@INJA A _17567) (@INJF A _17568))).
Proof. exact erefl. Qed.

Definition ZBOT {A : Type'} := @INJP A (@INJN A (NUMERAL 0)) (@ε (N -> A -> Prop) (fun z : N -> A -> Prop => True)).

Lemma ZBOT_def {A : Type'} : (@ZBOT A) = (@INJP A (@INJN A (NUMERAL N0)) (@ε (N -> A -> Prop) (fun z : N -> A -> Prop => True))).
Proof. exact erefl. Qed.

Inductive _ZRECSPACE {A : Type'} : (N -> A -> Prop) -> Prop :=
| ZRECSPACE0 : _ZRECSPACE ZBOT
| ZRECSPACE1 c i r : (forall n, _ZRECSPACE (r n)) -> _ZRECSPACE (ZCONSTR c i r).

Definition ZRECSPACE {A:Type'} := @_ZRECSPACE A.

Lemma ZRECSPACE_def {A : Type'} : (@ZRECSPACE A) = (fun a : N -> A -> Prop => forall ZRECSPACE' : (N -> A -> Prop) -> Prop, (forall a' : N -> A -> Prop, ((a' = (@ZBOT A)) \/ (exists c : N, exists i : A, exists r : N -> N -> A -> Prop, (a' = (@ZCONSTR A c i r)) /\ (forall n : N, ZRECSPACE' (r n)))) -> ZRECSPACE' a') -> ZRECSPACE' a).
Proof. ind_align. Qed.

Inductive recspace (A : Type) :=
| BOTTOM : recspace A
| CONSTR : N -> A -> (N -> recspace A) -> recspace A.

Arguments CONSTR [A] _ _ _.
Arguments BOTTOM {A}.

HB.instance Definition _ A := is_Type' (@BOTTOM A).

(* Explanations. *)

(* Suppose you wish to mutually define inductive types B1 ... Bn.

   - A is the product of the types of all external arguments in constructors of B1 ... Bn.
     ( Without waste, if two constructors use an argument of the same type, it won't appear twice in A. )
     ( Any argument not appearing in a constructor will be replaced by (ε (fun => True)). )
     ( If no constructor has external arguments then A is Prop by default, with only (ε (fun => True))
       appearing. )

   - N -> recspace A will contain all recursive arguments by emulating lists.
     ( Fnil and FCONS defined below emulate nil and cons. )
     ( Recursive arguments need to be either directly of type B_i for some i or of type T B_i where
       T is an already defined inductive type. In this case, HOL_Light adds another type
       to the mutual definition, TB_i (isomorphic to T B_i), maps T B_i to TB_i and then uses this mapping
       to define the actual constructors with arguments in T B_i. )

   - The first integer argument of CONSTR is used to index constructors.
     ( The first one defined will be assigned 0, the second 1, etc. )
 *)

(* Example of the definition of list A :
  - Defined with recspace A (for the one external argument in cons).
  - nil is the first constructor, so nil = CONSTR 0 (ε (fun _ => True)) Fnil.
  - cons is the second one, so cons a l = CONSTR 1 a (FCONS l Fnil). *)


Definition Fnil {A : Type} : N -> recspace A := fun _ => BOTTOM.

Definition FCONS {A : Type} (a : A) (f: N -> A) (n : N) : A :=
  N.recursion a (fun n _ => f n) n.

Notation "[ ]_rec" := Fnil (format "[ ]_rec").
Notation "[ x ]_rec" := (FCONS x Fnil).
Notation "[ x ; y ; .. ; z ]_rec" := (FCONS x (FCONS y .. (FCONS z Fnil) ..))
  (format "[ '[' x ; '/' y ; '/' .. ; '/' z ']' ]_rec").

Lemma FCONS_def {A : Type'} : @FCONS A = @ε ((prod N (prod N (prod N (prod N N)))) -> A -> (N -> A) -> N -> A) (fun FCONS' : (prod N (prod N (prod N (prod N N)))) -> A -> (N -> A) -> N -> A => forall _17460 : prod N (prod N (prod N (prod N N))), (forall a : A, forall f : N -> A, (FCONS' _17460 a f (NUMERAL N0)) = a) /\ (forall a : A, forall f : N -> A, forall n : N, (FCONS' _17460 a f (N.succ n)) = (f n))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))))).
Proof.
  by N_rec_align ; move=>* ; rewrite /FCONS N.recursion_succ.
Qed.

Fixpoint _dest_rec {A : Type'} (r : recspace A) : N -> A -> Prop :=
  match r with
  | BOTTOM => ZBOT
  | CONSTR n a f => ZCONSTR n a (fun m => _dest_rec (f m)) end.

Definition finv [A B : Type'] (f : A -> B) : B -> A := fun y => ε (fun x => f x = y).

Definition _mk_rec {A : Type'} : (N -> A -> Prop) -> recspace A := finv _dest_rec.

(* finv can define the inverse of a type embedding, the following will be useful to
   prove isomorphisms *)

Lemma finv_inv_l [A B : Type'] (f : A -> B) (x : A) :
  (forall x0 x1 : A, f x0 = f x1 -> x0 = x1) -> finv f (f x) = x.
Proof.
  intro H. apply H. apply (ε_spec (P := fun x' => f x' = f x)). now exists x. 
Qed.

Lemma finv_inv_r [A B : Type'] (f : A -> B) : forall (P : B -> Prop) (y : B), 
  (P y -> exists x, f x = y) -> ((exists x, f x = y) -> P y) -> P y = (f (finv f y) = y).
Proof.
  intros P y i1 i2. transitivity (exists x, f x = y).
  - exact (prop_ext i1 i2).
  - apply prop_ext;intro H.
    + exact (ε_spec H).
    + now exists (finv f y).
Qed.

Lemma axiom_10 : forall {A : Type'} (P : N -> A -> Prop), (@ZRECSPACE A P) = ((@_dest_rec A (@_mk_rec A P)) = P).
Proof.
  intros A P. apply finv_inv_r.
  - induction 1.
    + now exists BOTTOM.
    + exists (CONSTR c i (fun n => _mk_rec (r n))). simpl. f_equal. 
      ext 1 =>n. exact (ε_spec (H0 n)).
  - intros (r , <-). induction r ; now constructor.
Qed.

(* For axiom_9 we need to prove injectivity of _dest_rec, which requires to prove 
   injectivity of every embedding involved  *)
Lemma NUMSUM_INJ : forall b1 : Prop, forall x1 : N, forall b2 : Prop, forall x2 : N, ((NUMSUM b1 x1) = (NUMSUM b2 x2)) = ((b1 = b2) /\ (x1 = x2)).
Proof.
  intros b1 x1 b2 x2. apply prop_ext. 2: intros [e1 e2]; subst; reflexivity.
  unfold NUMSUM. do 2 if_intro.
  1,4 : split ; try now apply prop_ext.
  all : lia.
Qed.

Lemma INJN_INJ {A : Type'} : forall n1 : N, forall n2 : N, ((@INJN A n1) = (@INJN A n2)) = (n1 = n2).
Proof.
  intros n1 n2. apply prop_ext. 2: intro e; subst; reflexivity.
  intro e. generalize (ext_fun e n1); clear e; intro e.
  generalize (ext_fun e point); clear e. unfold INJN.
  by is_True (@erefl _ n1) => <-.
Qed.

Lemma INJA_INJ {A : Type'} : forall a1 : A, forall a2 : A, ((@INJA A a1) = (@INJA A a2)) = (a1 = a2).
Proof.
  intros a1 a2. apply prop_ext. 2: intro e; subst; reflexivity.
  intro e. generalize (ext_fun e 0); clear e; intro e.
  generalize (ext_fun e a2); clear e. unfold INJA.
  by rewrite (sym a1) ; is_True (@erefl _ a2) => ->.
Qed.

Lemma INJF_INJ {A : Type'} : forall f1 : N -> N -> A -> Prop, forall f2 : N -> N -> A -> Prop, ((@INJF A f1) = (@INJF A f2)) = (f1 = f2).
Proof.
  intros f1 f2. apply prop_ext. 2: intro e; subst; reflexivity.
  intro e. ext 3 => x y a.
  generalize (ext_fun e (NUMPAIR x y)); clear e; intro e.
  generalize (ext_fun e a); clear e. unfold INJF.
  rewrite NUMFST_NUMPAIR NUMSND_NUMPAIR. auto.
Qed.

Lemma Nodd_double n : N.odd (2 * n) = false.
Proof. rewrite N.odd_mul N.odd_2. reflexivity. Qed.

Lemma Neven_double n : N.even (2 * n) = true.
Proof. rewrite N.even_spec. exists n. reflexivity. Qed.

Lemma INJP_INJ {A : Type'} : forall f1 : N -> A -> Prop, forall f1' : N -> A -> Prop, forall f2 : N -> A -> Prop, forall f2' : N -> A -> Prop, ((@INJP A f1 f2) = (@INJP A f1' f2')) = ((f1 = f1') /\ (f2 = f2')).
Proof.
  intros f1 f1' f2 f2'. apply prop_ext. 2: intros [e1 e2]; subst; reflexivity.
  intro e.
  assert (e1: forall x a, INJP f1 f2 x a = INJP f1' f2' x a).
  intros x a. rewrite e. reflexivity.
  split; ext 2=> x a.
  generalize (e1 (N.succ (2 * x)) a). unfold INJP, NUMLEFT, NUMRIGHT.
  rewrite /COND N.even_succ Nodd_double !if_True succ_minus_1 NDIV_MULT ; auto. lia.
  generalize (e1 (2 * x) a). unfold INJP, NUMLEFT, NUMRIGHT.
  rewrite /COND Neven_double !if_False NDIV_MULT ; auto. lia.
Qed.

Lemma _dest_rec_inj {A : Type'} : 
  forall (r r' : recspace A), _dest_rec r = _dest_rec r' -> r = r'.
Proof.
  induction r ; induction r' ; simpl ;
  unfold ZBOT, ZCONSTR ; rewrite INJP_INJ ;
  rewrite INJN_INJ ; intros (e1, e2).
  - reflexivity.
  - destruct (N.neq_0_succ _ e1).
  - destruct (N.neq_succ_0 _ e1).
  - rewrite INJP_INJ INJF_INJ INJA_INJ in e2. f_equal.
    now apply N.succ_inj. now destruct e2.
    destruct e2 as (_ , e2). f_equal. ext=> m.
    apply H. exact (ext_fun e2 m).
Qed.

Lemma axiom_9 : forall {A : Type'} (a : recspace A), (@_mk_rec A (@_dest_rec A a)) = a.
Proof.
  intros A a. apply finv_inv_l. exact _dest_rec_inj.
Qed.

Lemma BOTTOM_def {A : Type'} : (@BOTTOM A) = (@_mk_rec A (@ZBOT A)).
Proof. symmetry. exact (axiom_9 BOTTOM). Qed.

Lemma CONSTR_def {A : Type'} : (@CONSTR A) = (fun _17591 : N => fun _17592 : A => fun _17593 : N -> recspace A => @_mk_rec A (@ZCONSTR A _17591 _17592 (fun n : N => @_dest_rec A (_17593 n)))).
Proof. symmetry. ext=>n a r. exact (axiom_9 (CONSTR n a r)). Qed.

Lemma FCONS_inj [A : Type'] (a a' : A) f f' : (FCONS a f = FCONS a' f') = (a = a' /\ f = f').
Proof.
  apply prop_ext;intro H. split. 
  - exact (ext_fun H 0).
  - ext =>n. generalize (ext_fun H (N.succ n)). unfold FCONS.
    now do 2 rewrite recursion_succ.
  - destruct H as (Ha , Hf). ext. apply N.peano_ind.
    + exact Ha.
    + intros n IHn. unfold FCONS. do 2 rewrite recursion_succ. now rewrite Hf.
Qed.

(*****************************************************************************)
(* Tactics to automatize inductive type alignment in most cases. *)
(*****************************************************************************)

(* In this section, we suppose that we wish to align a HOL_Light inductive definition to a
   Rocq one. In simple cases (Same ammount of constructors with same arguments,
   Rocq generates the correct inductive principle, currently no mutually defined types
   (check coq-hol-light-Logic for an example of alignment of 2 mutually defined types)), the following 
   tactics allow to fully automatize the proofs. *)

(* Let this also serve as a tutorial on how to map a HOL-Light type T in general.

   - Once file.ml has been translated with hol2dk,
     search for "Axiom" in file_terms.v. You should find axioms,
     usually named _mk_T and _dest_T, with type recspace A -> T and
     T -> recspace A respectively for some A.

   - In the mappings file, first define the correct Rocq inductive type if it does not exist,
     then define _dest_T yourself recursively.
     > To know how you should define it, look at the definitions after axioms _mk_T and _dest_t
       in T_terms.v. They should look like :
       "Definition _123456 := fun (...) => _mk_T [...]" where _123456 (for example) is simply a temporary name
       for a constructor C, replaced soon after with "Definition C := _123456".
       [_dest_T (C (...))] should then have value [...].

    - You can then define _mk_t := finv _dest_t.

    - Then, in file_types.v, you should find two axioms named axiom_... stating
      that _dest_T and _mk_t are the inverse of each other. Prove them in the mappings file using the following
      tactics :
      *)

(* _dest_inj proves that _dest is injective by double induction.
   Can fail when the induction principle isn't suited.
   Only works for non mutually defined types. *)
Ltac _dest_inj :=
  match goal with |- forall x x', _ => let e := fresh in
    induction x ; induction x' ; simpl ; intro e ;
    (* e is of the form "CONSTR n a f = CONSTR n' a' f'", so inversion
       gives hypotheses n=n' , a=a' and f=f'. *)
    inversion e ; auto ;
    repeat rewrite -> FCONS_inj in * ; (* f and f' should represent lists of recursive calls
                                       so we transform their equality into equality of
                                       each recursive call (so of the form
                                       "et : _dest_T t = _dest_T t'") *)
    f_equal ;
    match goal with IH : _ |- _ => now apply IH (* trying to apply the
                                                   induction hypothesis blindly to prove 
                                                   t = t' from et *)
    end end.

(* As long as _mk_T is defined as finv _dest_T, _mk_dest_rec will
   transform a goal of the form [forall x, (_mk_T (_dest_T x)) = x] into
   a goal stating injectivity of _dest_T thanks to finv_inv_l, then try to apply _dest_inj. *)
Ltac _mk_dest_rec :=
  intros ; apply finv_inv_l ;
  repeat match goal with useless : _ |- _ => clear useless end ; try _dest_inj.

(* Try to solve a goal of the form [forall r, P r = (_dest_T (_mk_T r)) = r)]
   for P defining the subset of recspace A with which the defined type(s) are
   isomorphism.
   Thanks to finv_inv_r, the goal can be replaced by [P r <-> exists x, _dest_T x = r]
   as long as _mk_T is defined as finv _dest_T.

   P is an inductive definition so the following is very similar to ind_align,
   except that [exists x, _dest_T x = r] is not inductive so we are rather inducting on x.
   Compared to ind_align, we do not have access to the constructor tactic to automatically
   find the correct constructor so it currently needs to be done by hand. *)
Ltac _dest_mk_rec :=
  let H := fresh in 
  let x := fresh "x" in 
  apply finv_inv_r ;
  [ intro H ; apply H ;
    clear H ; intros x H ;
    full_destruct ; rewrite H ;
    clear H ; simpl in *
  | let x := fresh "x" in
    (* simply inducting over [x] such that [_dest_ x = r]. *)
    intros (x,<-) ;
    induction x ; let P := fresh in
    let H' := fresh in
    intros P H' ; apply H' ; try breakgoal ].

(* - Finally, prove the definition of all constructors ( the lemmas _123456_def and C_def
     right under their definition in T_terms.v, replacing them with the new definition ).
     constr_align automatically proves _123456_def (afterwards, C_def is just reflexivity) : *)
Ltac extall := repeat (apply funext=>?).
Ltac constr_align H := (* Takes as argument the lemma [forall x, _mk_T (_dest_T x) = x].
                          Requires explicit type arguments. *)
  extall ; match goal with |- ?x = _ => exact (esym (H x)) end.

(****************************************************************************)
(* Some tactics to help automate recursive function alignments *)
(****************************************************************************)

(* Tries to prove a goal [f = ε P uv] where f is recursively defined. *)
Ltac total_align1 :=
  align_ε ; (* At this state, we have two goals : [P f] and [P f -> P f' -> f = f'].
                We now assume that [P f] is of the form
                [Q1 f C1 = ... /\ Q2 f C2 = ... /\ ... /\ Qn f Cn = ...]
                where the Ci are the constructors of the type and
                the Qi are universal quantifications over other arguments and subterms of the Ci. *)
  [ repeat split ; intros ; auto 
  | let f' := fresh in
    let r := fresh in
    let H := fresh in
    let H' := fresh in
    intros f' H H' ; ext 1 => r ; induction r ; extall ;
    try full_destruct ; (* with the correct induction principle, we have one case per clause,
                           we can replace [f] and [f']'s values with the corresponding
                           clause in [P] (that we have split).
                           By also rewriting all induction hypotheses,
                           the goal should become a reflexive equality.

                           For more complex types, it is possible to try and adapt this tactic
                           to specify how the induction hypothesis should be used.
                           See term_align in coq-hol-light-Logic1 for an example 
                           with lists as recursive arguments *)
    repeat match goal with
    H : _ |- _ => rewrite H end ;
    auto (* reflexivity would be more ideal but sometimes rewriting the induction hypothesis fails
            because the recursive call is dependant on something else, for example something quantified. *)
    ].

(* The following only change which argument induction is applied on. *)

Ltac total_align2 :=
  align_ε ; [ repeat split ; intros ; auto
  | let f' := fresh in
    let r := fresh in
    let a := fresh in
    let H := fresh in
    let H' := fresh in
    intros f' H H' ; ext 2=> a r ;
    revert a ; induction r ; intro a ; extall ;
    try full_destruct ; repeat match goal with
    H : ?P |- _ => rewrite H end ; auto ].

Ltac total_align3 :=
  align_ε ; [ repeat split ; intros ; auto
  | let f' := fresh in
    let r := fresh in
    let a := fresh in
    let b := fresh in
    let H := fresh in
    let H' := fresh in
    intros f' H H' ; ext 3 => a b r ;
    revert a b ; induction r ; intros a b ; extall ;
    try full_destruct ; repeat match goal with
    H : _ |- _ => rewrite H end ; auto ].

Ltac total_align4 :=
  align_ε ; [ repeat split ; intros ; auto
  | let f' := fresh in
    let r := fresh in
    let a := fresh in
    let b := fresh in
    let c := fresh in
    let H := fresh in
    let H' := fresh in
    intros f' H H' ; ext 4 => a b c r ;
    revert a b c ; induction r ; intros a b c ; extall ;
    try full_destruct ;
    repeat match goal with
    H : _ |- _ => rewrite H end ; auto ].

Ltac total_align5 :=
  align_ε ; [ repeat split ; intros ; auto
  | let f' := fresh in
    let r := fresh in
    let a := fresh in
    let b := fresh in
    let c := fresh in
    let d := fresh in
    let H := fresh in
    let H' := fresh in
    intros f' H H' ; ext 5 => a b c d r ;
    revert a b c d ; induction r ; intros a b c d ; extall ;
    try full_destruct ; repeat match goal with
    H : _ |- _ => rewrite H ; clear H end ; auto ].

Ltac total_align :=
  try total_align1 ;
  try total_align2 ;
  try total_align3 ;
  try total_align4 ;
  try total_align5.

(****************************************************************************)
(* Alignment of partial functions. *)
(****************************************************************************)

(* It is possible in HOL_Light to define a function
   recursively while not defining it for some constructors.
   The function will then have its value on these constructors chosen
   by the ε operator. In that case it is necessary to define the rocq function
   to be trivially equal to the HOL-Light one on each of these constructors.

   The following tactics allow to align such a partially defined function
   when provided with a predicate Q representing the cases where equality has to
   be trivial.

   Q should be defined inductively so as to be able to automatically discharge
   the goal [Q x -> _=_] via inversion. *)

(* First, the following lemmas mimick align_ε in the case where equality has to
   be trivial on Q. They can be used for any partial function, not just recursive
   (for example, a function defined through "new_specification") *)
Unset Implicit Arguments. 
Lemma partial_align_case1 {U A B : Type'} {uv0 : U} {x : A}
  (Q : A -> Prop) (f : U -> A -> B) (P : (U -> A -> B) -> Prop) :
  P f -> (forall x', Q x' -> f uv0 x' = ε P uv0 x') ->
  (forall f' uv x', P f ->  P f' -> (forall x'', Q x'' -> f uv x'' = f' uv x'') ->
  f uv x' = f' uv x') -> f uv0 x = ε P uv0 x.
Proof.
  intros Hf Htriv Hunique.
  apply Hunique;auto. apply ε_spec. now exists f.
Qed.

Lemma partial_align_case2 {U A B C : Type'} {uv0 : U} {x : B} {y : A}
  (Q : A -> Prop) (f : U -> B -> A -> C) (P : (U -> B -> A -> C) -> Prop) :
  P f -> (forall x' y', Q y' -> f uv0 x' y' = ε P uv0 x' y') ->
  (forall f' uv x' y', P f ->  P f' ->
  (forall x'' y'', Q y'' -> f uv x'' y'' = f' uv x'' y'') ->
  f uv x' y' = f' uv x' y') -> f uv0 x y = ε P uv0 x y.
Proof.
  intros Hf Htriv Hunique.
  apply Hunique;auto. apply ε_spec. now exists f.
Qed.

Lemma partial_align_case3 {U A B C D : Type'} {uv0 : U} {x : B} {y : C} {z : A}
  (Q : A -> Prop) (f : U -> B -> C -> A -> D) (P : (U -> B -> C -> A -> D) -> Prop) :
  P f -> (forall x' y' z', Q z' -> f uv0 x' y' z' = ε P uv0 x' y' z') ->
  (forall f' uv x' y' z', P f ->  P f' ->
  (forall x'' y'' z'', Q z'' -> f uv x'' y'' z'' = f' uv x'' y'' z'') ->
  f uv x' y' z' = f' uv x' y' z') -> f uv0 x y z = ε P uv0 x y z.
Proof.
  intros Hf Htriv Hunique.
  apply Hunique;auto. apply ε_spec. now exists f.
Qed.
Set Implicit Arguments.

(* The following ressembles total_align but also tries to automatically get rid of every cases that
   are in Q. It is designed for recursive functions only. *)
Ltac partial_align1 Q :=
  let f' := fresh "f'" in 
  let uv := fresh "uv" in
  let H := fresh in
  let H' := fresh "H'" in
  let Htriv := fresh "Htriv" in
  match goal with
  |- ?f ?x = ε _ _ ?x => apply (partial_align_case1 Q (fun _ => f)) ; (* replace f with (fun _ => f) uv *)
    clear x ; [repeat split ; auto
    | intro x ; now inversion 1 (* Additional goal
                                   [forall x, Q x -> f uv x = ε uv x] compared to
                                   total_align, if Q is inductive and the equality
                                   is trivial, inversion should do the job. *)
    | intros f' uv x H H' Htriv ; extall ;
      specialize (H uv) ; specialize (H' uv) ;
      induction x ; try (now apply Htriv ; try constructor ; auto) ; (* automatically takes care of cases
                                                                        in Q. *)
      clear Htriv ; (* We do not want to be able to rewrite Htriv outside of cases in Q. *)
      try full_destruct ;
      repeat match goal with H : _ |- _ => rewrite H end ; auto ] end.

Ltac partial_align2 Q :=
  let f' := fresh "f'" in 
  let uv := fresh "uv" in
  let H := fresh in
  let H' := fresh "H'" in
  let Htriv := fresh "Htriv" in
  match goal with
  |- ?f ?y ?x = ε _ _ ?y ?x => apply (partial_align_case2 Q (fun _ => f)) ;
    clear y x ; [repeat split ; auto
    | intros y x ; now inversion 1
    | intros f' uv y x H H' Htriv ; extall ;
      specialize (H uv) ; specialize (H' uv) ;
      induction x ; try (now apply Htriv ; try constructor ; auto) ;
      clear Htriv ; try full_destruct ;
      repeat match goal with H : _ |- _ => rewrite H end ; auto ] end.

Ltac partial_align3 Q :=
  let f' := fresh "f'" in 
  let uv := fresh "uv" in
  let H := fresh in
  let H' := fresh "H'" in
  let Htriv := fresh "Htriv" in
  match goal with
  |- ?f ?y ?z ?x = ε _ _ ?y ?z ?x => apply (partial_align_case3 Q (fun _ => f)) ;
    clear y z x ; [repeat split ; auto
    | intros y z x ; now inversion 1
    | intros f' uv y z x H H' Htriv ; extall ;
      specialize (H uv) ; specialize (H' uv) ;
      induction x ; try (now apply Htriv ; try constructor ; auto) ;
      clear Htriv ; try full_destruct ;
      repeat match goal with H : _ |- _ => rewrite H end ; auto ] end.

Ltac partial_align Q :=
  let x := fresh "x" in
  let y := fresh "y" in
  let z := fresh "z" in
  ext 1 => x ; partial_align1 Q +
  (ext 1 => y ; partial_align2 Q +
  (ext 1 => z ; partial_align3 Q)).

(****************************************************************************)
(* Alignment of the sum type constructor. *)
(****************************************************************************)

HB.instance Definition _ (A B : Type') := is_Type' (inl point : A+B).

Definition _dest_sum : forall {A B : Type'}, sum A B -> recspace (prod A B) :=
fun A B p => match p with
| inl a => CONSTR (NUMERAL N0) (a , ε (fun _ => True)) []_rec
| inr b => CONSTR (N.succ (NUMERAL N0)) (ε (fun _ => True) , b) []_rec
end.

Definition _mk_sum {A B : Type'} := finv (@_dest_sum A B).

Lemma axiom_11 {A B : Type'} : forall (a : sum A B), (@_mk_sum A B (@_dest_sum A B a)) = a.
Proof. _mk_dest_rec. Qed.

Lemma axiom_12 : forall {A B : Type'} (r : recspace (prod A B)), ((fun a : recspace (prod A B) => forall sum' : (recspace (prod A B)) -> Prop, (forall a' : recspace (prod A B), ((exists a'' : A, a' = ((fun a''' : A => @CONSTR (prod A B) (NUMERAL 0) (@pair A B a''' (@ε B (fun v : B => True))) (fun n : N => @BOTTOM (prod A B))) a'')) \/ (exists a'' : B, a' = ((fun a''' : B => @CONSTR (prod A B) (N.succ (NUMERAL N0)) (@pair A B (@ε A (fun v : A => True)) a''') (fun n : N => @BOTTOM (prod A B))) a''))) -> sum' a') -> sum' a) r) = ((@_dest_sum A B (@_mk_sum A B r)) = r).
Proof.
  intros A B r. _dest_mk_rec.
  now exists (inl x0). now exists (inr x0).
Qed.

Lemma INL_def {A B : Type'} : (@inl A B) = (fun a : A => @_mk_sum A B ((fun a' : A => @CONSTR (prod A B) (NUMERAL 0) (@pair A B a' (@ε B (fun v : B => True))) (fun n : N => @BOTTOM (prod A B))) a)).
Proof. constr_align (@axiom_11 A B). Qed.

Lemma INR_def {A B : Type'} : (@inr A B) = (fun a : B => @_mk_sum A B ((fun a' : B => @CONSTR (prod A B) (N.succ (NUMERAL N0)) (@pair A B (@ε A (fun v : A => True)) a') (fun n : N => @BOTTOM (prod A B))) a)).
Proof. constr_align (@axiom_11 A B). Qed.

(****************************************************************************)
(* Alignment of the option type constructor. *)
(****************************************************************************)

Definition _dest_option : forall {A : Type'}, option A -> recspace A :=
  fun A o =>
    match o with
    | None => CONSTR (NUMERAL N0) (ε (fun _ => True)) []_rec
    | Some a => CONSTR (N.succ (NUMERAL N0)) a []_rec
    end.

Definition _mk_option {A : Type'} := finv (@_dest_option A).

Lemma axiom_13 {A : Type'} : forall (a : option A), (@_mk_option A (@_dest_option A a)) = a.
Proof. _mk_dest_rec. Qed.

Definition option_pred {A : Type'} (r : recspace A) :=
  forall option' : recspace A -> Prop,
      (forall a' : recspace A,
       a' = CONSTR (NUMERAL N0) (ε (fun _ : A => True)) (fun _ : N => BOTTOM) \/
       (exists a'' : A, a' = CONSTR (N.succ (NUMERAL N0)) a'' (fun _ : N => BOTTOM)) ->
       option' a') -> option' r.

Lemma axiom_14' : forall {A : Type'} (r : recspace A), (option_pred r) = ((@_dest_option A (@_mk_option A r)) = r).
Proof.
  intros A r. _dest_mk_rec. now exists None. now exists (Some x0).
Qed.

Lemma axiom_14 : forall {A : Type'} (r : recspace A), ((fun a : recspace A => forall option' : (recspace A) -> Prop, (forall a' : recspace A, ((a' = (@CONSTR A (NUMERAL N0) (@ε A (fun v : A => True)) (fun n : N => @BOTTOM A))) \/ (exists a'' : A, a' = ((fun a''' : A => @CONSTR A (N.succ (NUMERAL N0)) a''' (fun n : N => @BOTTOM A)) a''))) -> option' a') -> option' a) r) = ((@_dest_option A (@_mk_option A r)) = r).
Proof. exact @axiom_14'. Qed.

Lemma NONE_def {A : Type'} : (@None A) = (@_mk_option A (@CONSTR A (NUMERAL N0) (@ε A (fun v : A => True)) (fun n : N => @BOTTOM A))).
Proof. constr_align (@axiom_13 A). Qed.

Lemma SOME_def {A : Type'} : (@Some A) = (fun a : A => @_mk_option A ((fun a' : A => @CONSTR A (N.succ (NUMERAL N0)) a' (fun n : N => @BOTTOM A)) a)).
Proof. constr_align (@axiom_13 A). Qed.

(****************************************************************************)
(* Alignment of the list type constructor. *)
(****************************************************************************)

HB.instance Definition _ A := is_Type' (@nil A).

Require Import Stdlib.Lists.List.

Fixpoint _dest_list {A : Type'} l : recspace A :=
  match l with
  | nil => CONSTR (NUMERAL N0) (ε (fun _ => True)) []_rec
  | a::l => CONSTR (N.succ (NUMERAL N0)) a [_dest_list l]_rec
  end.

Definition _mk_list {A : Type'} := finv (@_dest_list A).

Lemma axiom_15 {A : Type'} : forall (a : list A), (@_mk_list A (@_dest_list A a)) = a.
Proof. _mk_dest_rec. Qed.

Definition list_pred {A : Type'} (r : recspace A) :=
  forall list0 : recspace A -> Prop,
  (forall a' : recspace A,
  a' = CONSTR (NUMERAL N0) (ε (fun _ : A => True)) (fun _ : N => BOTTOM) \/
  (exists (a0 : A) (a1 : recspace A), a' = CONSTR (N.succ (NUMERAL N0)) a0 (FCONS a1 (fun _ : N => BOTTOM)) /\ list0 a1) -> list0 a')
  -> list0 r.

Lemma axiom_16' : forall {A : Type'} (r : recspace A), (list_pred r) = ((@_dest_list A (@_mk_list A r)) = r).
Proof.
  intros A r. _dest_mk_rec.
  - now exists nil.
  - exists (cons x0 x2). now rewrite <- H0.
  - right. exists a. exists (_dest_list x0). split.
    reflexivity. now apply IHx0.
Qed.

Lemma axiom_16 : forall {A : Type'} (r : recspace A), ((fun a : recspace A => forall list : (recspace A) -> Prop, (forall a' : recspace A, ((a' = (@CONSTR A (NUMERAL N0) (@ε A (fun v : A => True)) (fun n : N => @BOTTOM A))) \/ (exists a0 : A, exists a1 : recspace A, (a' = ((fun a0' : A => fun a1' : recspace A => @CONSTR A (N.succ (NUMERAL N0)) a0' (@FCONS (recspace A) a1' (fun n : N => @BOTTOM A))) a0 a1)) /\ (list a1))) -> list a') -> list a) r) = ((@_dest_list A (@_mk_list A r)) = r).
Proof. exact @axiom_16'. Qed.

Lemma NIL_def {A : Type'} : (@nil A) = (@_mk_list A (@CONSTR A (NUMERAL N0) (@ε A (fun v : A => True)) (fun n : N => @BOTTOM A))).
Proof. constr_align (@axiom_15 A). Qed.

Lemma CONS_def {A : Type'} : (@cons A) = (fun a0 : A => fun a1 : list A => @_mk_list A ((fun a0' : A => fun a1' : recspace A => @CONSTR A (N.succ (NUMERAL N0)) a0' (@FCONS (recspace A) a1' (fun n : N => @BOTTOM A))) a0 (@_dest_list A a1))).
Proof. constr_align (@axiom_15 A). Qed.

(****************************************************************************)
(* Tools to align some list functions *)
(****************************************************************************)

Inductive is_nil (A : Type) : list A -> Prop := nil_is_nil : is_nil nil.
Arguments is_nil : clear implicits.

Lemma if_list {A : Type} {B : Type'} {l : list A} {x y : B} : 
  (if l=nil then x else y) = match l with nil => x | _ => y end.
Proof.
  now if_intro ; destruct l.
Qed.

Ltac if_list := rewrite/COND if_list.

(****************************************************************************)
(* Alignment of list functions *)
(****************************************************************************)

Lemma APPEND_def {A : Type'} : (@app A) = (@ε ((prod N (prod N (prod N (prod N (prod N N))))) -> (list A) -> (list A) -> list A) (fun APPEND' : (prod N (prod N (prod N (prod N (prod N N))))) -> (list A) -> (list A) -> list A => forall _17935 : prod N (prod N (prod N (prod N (prod N N)))), (forall l : list A, (APPEND' _17935 (@nil A) l) = l) /\ (forall h : A, forall t : list A, forall l : list A, (APPEND' _17935 (@cons A h t) l) = (@cons A h (APPEND' _17935 t l)))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))))))))).
Proof.
  total_align.
Qed.

Lemma REVERSE_def {A : Type'} : (@rev A) = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> (list A) -> list A) (fun REVERSE' : (prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> (list A) -> list A => forall _17939 : prod N (prod N (prod N (prod N (prod N (prod N N))))), ((REVERSE' _17939 (@nil A)) = (@nil A)) /\ (forall l : list A, forall x : A, (REVERSE' _17939 (@cons A x l)) = (@app A (REVERSE' _17939 l) (@cons A x (@nil A))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))))))))).
Proof.
  total_align.
Qed.

Fixpoint lengthN {A : Type} (l : list A) :=
match l with
|nil => N0
|_::l => N.succ (lengthN l) end.

(* in case it might be useful ? *)
Lemma length_of_nat_N {A : Type} (l : list A) : N.of_nat (length l) = lengthN l.
Proof.
  induction l. auto. unfold length. rewrite Nnat.Nat2N.inj_succ. simpl.
  now rewrite <- IHl.
Qed.

Lemma LENGTH_def {A : Type'} : lengthN = (@ε ((prod N (prod N (prod N (prod N (prod N N))))) -> (list A) -> N) (fun LENGTH' : (prod N (prod N (prod N (prod N (prod N N))))) -> (list A) -> N => forall _18106 : prod N (prod N (prod N (prod N (prod N N)))), ((LENGTH' _18106 (@nil A)) = N0) /\ (forall h : A, forall t : list A, (LENGTH' _18106 (@cons A h t)) = (N.succ (LENGTH' _18106 t)))) (@pair N (prod N (prod N (prod N (prod N N)))) ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) ((BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) ((BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) ((BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))))))))).
Proof.
  total_align.
Qed.

Lemma MAP_def {A B : Type'} : (@map A B) = (@ε ((prod N (prod N N)) -> (A -> B) -> (list A) -> list B) (fun MAP' : (prod N (prod N N)) -> (A -> B) -> (list A) -> list B => forall _17950 : prod N (prod N N), (forall f : A -> B, (MAP' _17950 f (@nil A)) = (@nil B)) /\ (forall f : A -> B, forall h : A, forall t : list A, (MAP' _17950 f (@cons A h t)) = (@cons B (f h) (MAP' _17950 f t)))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))).
Proof.
 total_align.
Qed.

Lemma BUTLAST_def {_25251 : Type'} : (@removelast _25251) = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> (list _25251) -> list _25251) (fun BUTLAST' : (prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> (list _25251) -> list _25251 => forall _17958 : prod N (prod N (prod N (prod N (prod N (prod N N))))), ((BUTLAST' _17958 (@nil _25251)) = (@nil _25251)) /\ (forall h : _25251, forall t : list _25251, (BUTLAST' _17958 (@cons _25251 h t)) = (@COND (list _25251) (t = (@nil _25251)) (@nil _25251) (@cons _25251 h (BUTLAST' _17958 t))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0))))))))))))))).
Proof.
  total_align. now if_list.
Qed.

Lemma ALL_def {_25307 : Type'} : (@Forall _25307) = (@ε ((prod N (prod N N)) -> (_25307 -> Prop) -> (list _25307) -> Prop) (fun ALL' : (prod N (prod N N)) -> (_25307 -> Prop) -> (list _25307) -> Prop => forall _17973 : prod N (prod N N), (forall P : _25307 -> Prop, (ALL' _17973 P (@nil _25307)) = True) /\ (forall h : _25307, forall P : _25307 -> Prop, forall t : list _25307, (ALL' _17973 P (@cons _25307 h t)) = ((P h) /\ (ALL' _17973 P t)))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))))))).
Proof.
  total_align;intros.
  - rewrite is_True. apply Forall_nil.
  - apply prop_ext;intro H. now inversion H. destruct H. now apply Forall_cons.
Qed.

Lemma PAIRWISE_def {A : Type'} : (@ForallOrdPairs A) = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> (A -> A -> Prop) -> (list A) -> Prop) (fun PAIRWISE' : (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> (A -> A -> Prop) -> (list A) -> Prop => forall _18057 : prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))), (forall r : A -> A -> Prop, (PAIRWISE' _18057 r (@nil A)) = True) /\ (forall h : A, forall r : A -> A -> Prop, forall t : list A, (PAIRWISE' _18057 r (@cons A h t)) = ((@Forall A (r h) t) /\ (PAIRWISE' _18057 r t)))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N N)))))) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))))))))))).
Proof.
  total_align;intros.
  - rewrite is_True. apply FOP_nil.
  - apply prop_ext;intro H. now inversion H. destruct H. now apply FOP_cons.
Qed.

Notation "'@FILTER'" := (@List.filter : forall A : Type, (A -> Prop) -> list A -> list A) (only parsing).

Lemma _FILTER_def {A : Type'} : (@FILTER A) = (@ε ((prod N (prod N (prod N (prod N (prod N N))))) -> (A -> Prop) -> (list A) -> list A) (fun FILTER' : (prod N (prod N (prod N (prod N (prod N N))))) -> (A -> Prop) -> (list A) -> list A => forall _18185 : prod N (prod N (prod N (prod N (prod N N)))), (forall P : A -> Prop, (FILTER' _18185 P (@nil A)) = (@nil A)) /\ (forall h : A, forall P : A -> Prop, forall t : list A, (FILTER' _18185 P (@cons A h t)) = (@COND (list A) (P h) (@cons A h (FILTER' _18185 P t)) (FILTER' _18185 P t)))) (@pair N (prod N (prod N (prod N (prod N N)))) ((BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) ((BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) ((BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))))))))).
Proof.
  total_align.
Qed.

Lemma MEM_def {_25376 : Type'} : (@In _25376) = (@ε ((prod N (prod N N)) -> _25376 -> (list _25376) -> Prop) (fun MEM' : (prod N (prod N N)) -> _25376 -> (list _25376) -> Prop => forall _17995 : prod N (prod N N), (forall x : _25376, (MEM' _17995 x (@nil _25376)) = False) /\ (forall h : _25376, forall x : _25376, forall t : list _25376, (MEM' _17995 x (@cons _25376 h t)) = ((x = h) \/ (MEM' _17995 x t)))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0))))))))))).
Proof.
  total_align. now rewrite (sym x h).
Qed.

Fixpoint repeatpos {A : Type} (n : positive) (a : A) : list A :=
match n with
|xH => a::nil
|xO n => let l := repeatpos n a in l++l
|xI n => let l := repeatpos n a in a::l++l end. 

Definition repeatN {A : Type} (n : N) (a : A) : list A :=
match n with
|0 => nil
|Npos n => repeatpos n a end.

Lemma repeatpos_sym {A : Type} (a : A) (p p' : positive) :
  repeatpos p a ++ a :: repeatpos p' a = a :: repeatpos p a ++ repeatpos p' a.
Proof. 
  revert p'. induction p;intro p'.
  - simpl. rewrite <- app_assoc. rewrite IHp.
    do 3 rewrite app_comm_cons. rewrite <- IHp.
    repeat rewrite app_comm_cons. now rewrite app_assoc.
  - simpl. rewrite <- app_assoc. rewrite IHp.
    do 2 rewrite app_comm_cons. rewrite <- IHp.
    repeat rewrite app_comm_cons. now rewrite app_assoc.
  - auto.
Qed.

Lemma repeatN_succ {A : Type} n (a : A) : repeatN (N.succ n) a = a :: repeatN n a.
Proof.
  destruct n. auto.
  induction p;auto.
  - simpl in *. repeat rewrite IHp. rewrite <- app_comm_cons.
    now rewrite repeatpos_sym.
Qed.

(* In case it is needed *)
Lemma repeat_to_nat_N {A : Type} n (a : A) :
  repeat a (N.to_nat n) = repeatN n a.
Proof.
  revert n. apply N.peano_ind.
  reflexivity. intros n IHn.
  now rewrite repeatN_succ Nnat.N2Nat.inj_succ -IHn.
Qed.

Lemma REPLICATE_def {A : Type'} : repeatN = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))))) -> N -> A -> list A) (fun REPLICATE' : (prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))))) -> N -> A -> list A => forall _18125 : prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))), (forall x : A, (REPLICATE' _18125 N0 x) = (@nil A)) /\ (forall n : N, forall x : A, (REPLICATE' _18125 (N.succ n) x) = (@cons A x (REPLICATE' _18125 n x)))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) ((BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N N)))))) ((BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) ((BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) ((BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) ((BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) ((BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0))))))))))))))))).
Proof.
  N_rec_align. exact repeatN_succ.
Qed.

Definition fold_right_with_perm_args {A B : Type'} 
(f: A -> B -> B) (l: list A) (b: B) : B := @fold_right B A f b l.

Lemma ITLIST_def {A B : Type'} : (@fold_right_with_perm_args A B) = (@ε ((prod N (prod N (prod N (prod N (prod N N))))) -> (A -> B -> B) -> (list A) -> B -> B) (fun ITLIST' : (prod N (prod N (prod N (prod N (prod N N))))) -> (A -> B -> B) -> (list A) -> B -> B => forall _18151 : prod N (prod N (prod N (prod N (prod N N)))), (forall f : A -> B -> B, forall b : B, (ITLIST' _18151 f (@nil A) b) = b) /\ (forall h : A, forall f : A -> B -> B, forall t : list A, forall b : B, (ITLIST' _18151 f (@cons A h t) b) = (f h (ITLIST' _18151 f t b)))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))))))))).
Proof.
  total_align.
Qed.

Definition HD {A : Type'} := @ε ((prod N N) -> (list A) -> A) (fun HD' : (prod N N) -> (list A) -> A => forall _17927 : prod N N, forall t : list A, forall h : A, (HD' _17927 (@cons A h t)) = h) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 0))))))))).

Definition hd {A:Type'} := @hd A (HD nil).

Lemma HD_def {A : Type'} : @hd A = @HD A.
Proof. unfold HD. partial_align (is_nil A). Qed.

Definition TL {A : Type'} := (@ε ((prod N N) -> (list A) -> list A) (fun TL' : (prod N N) -> (list A) -> list A => forall _17931 : prod N N, forall h : A, forall t : list A, (TL' _17931 (@cons A h t)) = t) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 0)))))))))).

Definition tl {A : Type'} (l : list A) :=
match l with
| nil => @TL A nil
| cons h t => @tl A (cons h t)
end.

Lemma TL_def {A : Type'} : @tl A = @TL A.
Proof.
  unfold TL. partial_align (is_nil A).
Qed.

Lemma NULL_def {A : Type'} : is_nil A = (@ε ((prod N (prod N (prod N N))) -> (list A) -> Prop) (fun NULL' : (prod N (prod N (prod N N))) -> (list A) -> Prop => forall _18129 : prod N (prod N (prod N N)), ((NULL' _18129 (@nil A)) = True) /\ (forall h : A, forall t : list A, (NULL' _18129 (@cons A h t)) = False)) (@pair N (prod N (prod N N)) ((BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))))))).
Proof.
  total_align. rewrite is_True. exists. rewrite is_False. inversion 1.
Qed.

Lemma EX_def {A : Type'} : @Exists A = (@ε ((prod N N) -> (A -> Prop) -> (list A) -> Prop) (fun EX' : (prod N N) -> (A -> Prop) -> (list A) -> Prop => forall _18143 : prod N N, (forall P : A -> Prop, (EX' _18143 P (@nil A)) = False) /\ (forall h : A, forall P : A -> Prop, forall t : list A, (EX' _18143 P (@cons A h t)) = ((P h) \/ (EX' _18143 P t)))) (@pair N N ((BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) ((BIT0 (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 N0)))))))))).
Proof.
  total_align;intros.
  - rewrite is_False. intro H. inversion H.
  - apply prop_ext;intro H. inversion H;auto.
    destruct H. now apply Exists_cons_hd. now apply Exists_cons_tl.
Qed.

Lemma ALL2_def {A B : Type'} : @Forall2 A B = (@ε ((prod N (prod N (prod N N))) -> (A -> B -> Prop) -> (list A) -> (list B) -> Prop) (fun ALL2' : (prod N (prod N (prod N N))) -> (A -> B -> Prop) -> (list A) -> (list B) -> Prop => forall _18166 : prod N (prod N (prod N N)), (forall P : A -> B -> Prop, forall l2 : list B, (ALL2' _18166 P (@nil A) l2) = (l2 = (@nil B))) /\ (forall h1' : A, forall P : A -> B -> Prop, forall t1 : list A, forall l2 : list B, (ALL2' _18166 P (@cons A h1' t1) l2) = (@COND Prop (l2 = (@nil B)) False ((P h1' (@hd B l2)) /\ (ALL2' _18166 P t1 (@tl B l2)))))) (@pair N (prod N (prod N N)) ((BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) ((BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT1 N0))))))))))).
Proof.
  total_align.
  - apply prop_ext;intro H. now inversion H. rewrite H. apply Forall2_nil.
  - if_list. apply prop_ext;intro. now inversion H. induction l2.
    destruct H. now apply Forall2_cons.
Qed.

Definition LAST {A : Type'} := (@ε ((prod N (prod N (prod N N))) -> (list A) -> A) (fun LAST' : (prod N (prod N (prod N N))) -> (list A) -> A => forall _18117 : prod N (prod N (prod N N)), forall h : A, forall t : list A, (LAST' _18117 (@cons A h t)) = (@COND A (t = (@nil A)) h (LAST' _18117 t))) (@pair N (prod N (prod N N)) ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) ((BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))))))).

Definition last {A : Type'} (l : list A) := last l (LAST nil).

Lemma LAST_def {A : Type'} : last = (@ε ((prod N (prod N (prod N N))) -> (list A) -> A) (fun LAST' : (prod N (prod N (prod N N))) -> (list A) -> A => forall _18117 : prod N (prod N (prod N N)), forall h : A, forall t : list A, (LAST' _18117 (@cons A h t)) = (@COND A (t = (@nil A)) h (LAST' _18117 t))) (@pair N (prod N (prod N N)) ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) ((BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))))))).
Proof.
  partial_align (is_nil A). intros. now if_list.
Qed.

Fixpoint map2 {A B C : Type'} (f : A -> B -> C) (l : list A) (l' : list B) : list C := 
match l with 
|nil => nil 
|a::l => f a (hd l') :: map2 f l (tl l') end.

Lemma MAP2_def {A B C : Type'} : map2 = (@ε ((prod N (prod N (prod N N))) -> (A -> B -> C) -> (list A) -> (list B) -> list C) (fun MAP2' : (prod N (prod N (prod N N))) -> (A -> B -> C) -> (list A) -> (list B) -> list C => forall _18174 : prod N (prod N (prod N N)), (forall f : A -> B -> C, forall l : list B, (MAP2' _18174 f (@nil A) l) = (@nil C)) /\ (forall h1' : A, forall f : A -> B -> C, forall t1 : list A, forall l : list B, (MAP2' _18174 f (@cons A h1' t1) l) = (@cons C (f h1' (@hd B l)) (MAP2' _18174 f t1 (@tl B l))))) (@pair N (prod N (prod N N)) ((BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) ((BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT1 N0))))))))))).
Proof.
  total_align.
Qed.

(* Cannot align with nth : different possible default values *)
Fixpoint nth_partial {A : Type'} n (l : list A) :=
  match n with
  | O => hd l
  | S n => nth_partial n (tl l) end.

Definition Nth {A : Type'} n (l : list A) := nth_partial (N.to_nat n) l.

(* In case it's needed *)
Lemma nth_to_nat_N {A : Type'} :
  forall n (l : list A) default, (length l > n)%coq_nat -> nth_partial n l = nth n l default.
Proof.
  intros n l default H. revert l H. induction n ; intros l H ; destruct l.
  1,3 : destruct (Nat.nlt_0_r _ H).
  - reflexivity.
  - apply IHn. now apply PeanoNat.lt_S_n.
Qed.

Lemma EL_def {A : Type'} : Nth = (@ε ((prod N N) -> N -> (list A) -> A) (fun EL' : (prod N N) -> N -> (list A) -> A => forall _18178 : prod N N, (forall l : list A, (EL' _18178 N0 l) = (@hd A l)) /\ (forall n : N, forall l : list A, (EL' _18178 (N.succ n) l) = (EL' _18178 n (@tl A l)))) (@pair N N ((BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))))).
Proof.
  N_rec_align. intros n l. unfold Nth. now rewrite Nnat.N2Nat.inj_succ.
Qed.

Definition ASSOC {A B : Type'} := (@ε ((prod N (prod N (prod N (prod N N)))) -> A -> (list (prod A B)) -> B) (fun ASSOC' : (prod N (prod N (prod N (prod N N)))) -> A -> (list (prod A B)) -> B => forall _18192 : prod N (prod N (prod N (prod N N))), forall h : prod A B, forall a : A, forall t : list (prod A B), (ASSOC' _18192 a (@cons (prod A B) h t)) = (@COND B ((@fst A B h) = a) (@snd A B h) (ASSOC' _18192 a t))) (@pair N (prod N (prod N (prod N N))) ((BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) ((BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) ((BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 N0))))))))))))).

Fixpoint assoc {A B : Type'} (a : A) (l : list (A * B)) := 
match l with
|nil => ASSOC a nil
|c::l => if fst c = a then snd c else assoc a l end.

Lemma ASSOC_def {A B : Type'} : assoc = (@ε ((prod N (prod N (prod N (prod N N)))) -> A -> (list (prod A B)) -> B) (fun ASSOC' : (prod N (prod N (prod N (prod N N)))) -> A -> (list (prod A B)) -> B => forall _18192 : prod N (prod N (prod N (prod N N))), forall h : prod A B, forall a : A, forall t : list (prod A B), (ASSOC' _18192 a (@cons (prod A B) h t)) = (@COND B ((@fst A B h) = a) (@snd A B h) (ASSOC' _18192 a t))) (@pair N (prod N (prod N (prod N N))) ((BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) ((BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) ((BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 N0))))))))))))).
Proof. partial_align (is_nil (A*B)). Qed.

Fixpoint zip {A B : Type'} (l : list A) (l' : list B) := 
match l with
|nil => nil
|a::l => (a,hd l') :: zip l (tl l') end.

(* In case it is needed *)
Lemma zip_combine {A B : Type'} (l : list A) (l' : list B) :
  (length l <= length l')%coq_nat -> zip l l' = combine l l'.
Proof.
  revert l'. induction l ; intros l' H.
  reflexivity. destruct l'.
  - simpl in H. destruct (Nat.nle_succ_0 _ H).
  - simpl. rewrite IHl. reflexivity. now apply le_S_n.
Qed.

Lemma ZIP_def {A B : Type'} : zip = (@ε ((prod N (prod N N)) -> (list A) -> (list B) -> list (prod A B)) (fun ZIP' : (prod N (prod N N)) -> (list A) -> (list B) -> list (prod A B) => forall _18205 : prod N (prod N N), (forall l2 : list B, (ZIP' _18205 (@nil A) l2) = (@nil (prod A B))) /\ (forall h1' : A, forall t1 : list A, forall l2 : list B, (ZIP' _18205 (@cons A h1' t1) l2) = (@cons (prod A B) (@pair A B h1' (@hd B l2)) (ZIP' _18205 t1 (@tl B l2))))) (@pair N (prod N N) ((BIT0 (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) ((BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0))))))))))).
Proof.
  total_align.
Qed.

Fixpoint Forallpairs {A B : Type} (f : A -> B -> Prop) (l : list A) (l' : list B) := 
match l with
|nil => True
|a::l => Forall (f a) l' /\ Forallpairs f l l' end.

Lemma ALLPAIRS_def {A B : Type'} : Forallpairs = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> (A -> B -> Prop) -> (list A) -> (list B) -> Prop) (fun ALLPAIRS' : (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> (A -> B -> Prop) -> (list A) -> (list B) -> Prop => forall _18213 : prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))), (forall f : A -> B -> Prop, forall l : list B, (ALLPAIRS' _18213 f (@nil A) l) = True) /\ (forall h : A, forall f : A -> B -> Prop, forall t : list A, forall l : list B, (ALLPAIRS' _18213 f (@cons A h t) l) = ((@List.Forall B (f h) l) /\ (ALLPAIRS' _18213 f t l)))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N N)))))) ((BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) ((BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) ((BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) ((BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))))))))))).
Proof.
  total_align.
Qed.

Definition list_of_Nseq {A : Type'} := 
let fix list_of_seq (s : N -> A) (n : nat) :=
match n with
|O => nil
|S n => list_of_seq s n ++ (s (N.of_nat n) :: nil) end in
fun s n => list_of_seq s (N.to_nat n).

Lemma list_of_seq_def {A : Type'} : list_of_Nseq = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))))))) -> (N -> A) -> N -> list A) (fun list_of_seq' : (prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))))))) -> (N -> A) -> N -> list A => forall _18227 : prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))))), (forall s : N -> A, (list_of_seq' _18227 s N0) = (@nil A)) /\ (forall s : N -> A, forall n : N, (list_of_seq' _18227 s (N.succ n)) = (@app A (list_of_seq' _18227 s n) (@cons A (s n) (@nil A))))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))))) ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))))) ((BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) ((BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N N)))))) ((BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) ((BIT1 (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) ((BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) ((BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) ((BIT1 (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 N0)))))))) (@pair N N ((BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT1 N0)))))))) ((BIT1 (BIT0 (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 N0))))))))))))))))))).
Proof.
  N_rec_align. intros s n. unfold list_of_Nseq.
  rewrite Nnat.N2Nat.inj_succ. now rewrite Nnat.N2Nat.id. 
Qed.

Fixpoint fold_right2 {A B C : Type'} (f : A -> B -> C -> C) 
(l : list A) (l' : list B) (c : C) : C := 
match l with 
|nil => c 
|a::l => (f a (hd l') (fold_right2 f l (tl l') c)) end.

Lemma ITLIST2_def {A B C : Type'} : fold_right2 = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> (A -> B -> C -> C) -> (list A) -> (list B) -> C -> C) (fun ITLIST2' : (prod N (prod N (prod N (prod N (prod N (prod N N)))))) -> (A -> B -> C -> C) -> (list A) -> (list B) -> C -> C => forall _18201 : prod N (prod N (prod N (prod N (prod N (prod N N))))), (forall f : A -> B -> C -> C, forall l2 : list B, forall b : C, (ITLIST2' _18201 f (@nil A) l2 b) = b) /\ (forall h1' : A, forall f : A -> B -> C -> C, forall t1 : list A, forall l2 : list B, forall b : C, (ITLIST2' _18201 f (@cons A h1' t1) l2 b) = (f h1' (@hd B l2) (ITLIST2' _18201 f t1 (@tl B l2) b)))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) ((BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) ((BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) ((BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) ((BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) ((BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N ((BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) ((BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT1 N0)))))))))))))).
Proof.
  total_align.
Qed.

(****************************************************************************)
(* Alignment of the type of ASCII characters. *)
(****************************************************************************)

(* Note the mismatch between Rocq's ascii which takes booleans as arguments
and HOL-Light's char which takes propositions as arguments. *)

Require Import Stdlib.Strings.Ascii.

HB.instance Definition _ := is_Type' zero.

Definition _dest_char : ascii -> recspace (Prop*(Prop*(Prop*(Prop*(Prop*(Prop*(Prop*(Prop)))))))) :=
  fun a => match a with
  | Ascii a0 a1 a2 a3 a4 a5 a6 a7 => CONSTR 0
    ((fun a0 a1 a2 a3 a4 a5 a6 a7 : Prop => (a0,(a1,(a2,(a3,(a4,(a5,(a6,(a7)))))))))
    a0 a1 a2 a3 a4 a5 a6 a7) []_rec end.

Definition _mk_char := finv _dest_char.

Lemma axiom_17 : forall (a : ascii), (_mk_char (_dest_char a)) = a.
Proof.
  by _mk_dest_rec; intros [] [] [=] * ; f_equal ; AllProp.
Qed.

Definition char_pred (r : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))) :=
  forall char' : (recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))) -> Prop, (forall a' : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))), (exists a0 : Prop, exists a1 : Prop, exists a2 : Prop, exists a3 : Prop, exists a4 : Prop, exists a5 : Prop, exists a6 : Prop, exists a7 : Prop, a' =
    ((fun a0' : Prop => fun a1' : Prop => fun a2' : Prop => fun a3' : Prop => fun a4' : Prop => fun a5' : Prop => fun a6' : Prop => fun a7' : Prop => @CONSTR (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))) (NUMERAL N0) (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))) a0' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))) a1' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))) a2' (@pair Prop (prod Prop (prod Prop (prod Prop Prop))) a3' (@pair Prop (prod Prop (prod Prop Prop)) a4' (@pair Prop (prod Prop Prop) a5' (@pair Prop Prop a6' a7'))))))) (fun n : N => @BOTTOM (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))))) a0 a1 a2 a3 a4 a5 a6 a7)) -> char' a') -> char' r.

Inductive char_ind : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))) -> Prop :=
| char_ind_cons a0 a1 a2 a3 a4 a5 a6 a7 :
  char_ind (CONSTR (NUMERAL N0) (is_true a0, (is_true a1, (is_true a2, (is_true a3, (is_true a4, (is_true a5, (is_true a6, is_true a7))))))) (fun _ => BOTTOM)).

Lemma axiom_18' : forall (r : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))),
char_pred r = ((_dest_char (_mk_char r)) = r).
Proof.
  move=>r; _dest_mk_rec.
  - by exists (Ascii x0 x1 x2 x3 x4 x5 x6 x7)=> /= ; repeat rewrite asboolE.
  - by exist b b0 b1 b2 b3 ; exist b4 b5 b6.
Qed.

Lemma axiom_18 : forall (r : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))), ((fun a : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))) => forall char' : (recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))))) -> Prop, (forall a' : recspace (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))), (exists a0 : Prop, exists a1 : Prop, exists a2 : Prop, exists a3 : Prop, exists a4 : Prop, exists a5 : Prop, exists a6 : Prop, exists a7 : Prop, a' =
((fun a0' : Prop => fun a1' : Prop => fun a2' : Prop => fun a3' : Prop => fun a4' : Prop => fun a5' : Prop => fun a6' : Prop => fun a7' : Prop => @CONSTR (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))) (NUMERAL N0) (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))) a0' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))) a1' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))) a2' (@pair Prop (prod Prop (prod Prop (prod Prop Prop))) a3' (@pair Prop (prod Prop (prod Prop Prop)) a4' (@pair Prop (prod Prop Prop) a5' (@pair Prop Prop a6' a7'))))))) (fun n : N => @BOTTOM (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))))) a0 a1 a2 a3 a4 a5 a6 a7)) -> char' a') -> char' a) r) = ((_dest_char (_mk_char r)) = r).
Proof. exact axiom_18'. Qed.

(*****************************************************************************)
(* Alignment of the type nadd of nearly additive sequences of naturals. *)
(*****************************************************************************)

Definition dist := fun p : prod N N => N.add (N.sub (@fst N N p) (@snd N N p)) (N.sub (@snd N N p) (@fst N N p)).

Lemma dist_def : dist = (fun _22947 : prod N N => N.add (N.sub (@fst N N _22947) (@snd N N _22947)) (N.sub (@snd N N _22947) (@fst N N _22947))).
Proof. exact erefl. Qed.

Lemma DIST_REFL : forall n : N, dist (n,n) = 0.
Proof. intro n. unfold dist. simpl. rewrite N.sub_diag. reflexivity. Qed.

Lemma DIST_SYM x y : dist (x,y) = dist (y,x).
Proof. unfold dist; simpl. lia. Qed.

Lemma DIST_TRIANGLE x y z : dist (x,z) <= dist (x,y) + dist (y,z).
Proof. unfold dist; simpl. lia. Qed.

Definition is_nadd := fun f : N -> N => exists B : N, forall m : N, forall n : N, N.le (dist (@pair N N (N.mul m (f n)) (N.mul n (f m)))) (N.mul B (N.add m n)).

Lemma is_nadd_def : is_nadd = (fun _23257 : N -> N => exists B : N, forall m : N, forall n : N, N.le (dist (@pair N N (N.mul m (_23257 n)) (N.mul n (_23257 m)))) (N.mul B (N.add m n))).
Proof. exact erefl. Qed.

Lemma is_nadd_times n : is_nadd (fun x => n * x).
Proof.
  exists 0. intros x y. simpl. assert (e: x*(n*y)=y*(n*x)). lia.
  rewrite e DIST_REFL. reflexivity.
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

Lemma nadd_of_num_def : nadd_of_num = (fun _23288 : N => mk_nadd (fun n : N => N.mul _23288 n)).
Proof. exact erefl. Qed.

Definition nadd_le : nadd -> nadd -> Prop := fun _23295 : nadd => fun _23296 : nadd => exists B : N, forall n : N, N.le (dest_nadd _23295 n) (N.add (dest_nadd _23296 n) B).

Lemma nadd_le_def : nadd_le = (fun _23295 : nadd => fun _23296 : nadd => exists B : N, forall n : N, N.le (dest_nadd _23295 n) (N.add (dest_nadd _23296 n) B)).
Proof. exact erefl. Qed.

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

Lemma nadd_add_def : nadd_add = (fun _23311 : nadd => fun _23312 : nadd => mk_nadd (fun n : N => N.add (dest_nadd _23311 n) (dest_nadd _23312 n))).
Proof. exact erefl. Qed.

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
  unfold nadd_add, nadd_of_num. f_equal. ext=> x.
  rewrite axiom_20_aux. 2: apply is_nadd_times.
  rewrite axiom_20_aux. 2: apply is_nadd_times.
  lia.
Qed.

Lemma NADD_ADD_SYM p q : nadd_add p q = nadd_add q p.
Proof. unfold nadd_add. f_equal. ext=>x. lia. Qed.

Lemma NADD_ADD_ASSOC p q r :
  nadd_add (nadd_add p q) r = nadd_add p (nadd_add q r).
Proof.
  unfold nadd_add. f_equal. ext=>x. rewrite !axiom_20_aux. lia.
  apply is_nadd_add. apply is_nadd_add.
Qed.

Definition nadd_mul : nadd -> nadd -> nadd := fun _23325 : nadd => fun _23326 : nadd => mk_nadd (fun n : N => dest_nadd _23325 (dest_nadd _23326 n)).

Lemma nadd_mul_def : nadd_mul = (fun _23325 : nadd => fun _23326 : nadd => mk_nadd (fun n : N => dest_nadd _23325 (dest_nadd _23326 n))).
Proof. exact erefl. Qed.

Definition nadd_rinv : nadd -> N -> N := fun _23462 : nadd => fun n : N => N.div (N.mul n n) (dest_nadd _23462 n).

Lemma nadd_rinv_def : nadd_rinv = (fun _23462 : nadd => fun n : N => N.div (N.mul n n) (dest_nadd _23462 n)).
Proof. exact erefl. Qed.

Definition nadd_eq : nadd -> nadd -> Prop := fun _23276 : nadd => fun _23277 : nadd => exists B : N, forall n : N, N.le (dist (@pair N N (dest_nadd _23276 n) (dest_nadd _23277 n))) B.

Lemma nadd_eq_def : nadd_eq = (fun _23276 : nadd => fun _23277 : nadd => exists B : N, forall n : N, N.le (dist (@pair N N (dest_nadd _23276 n) (dest_nadd _23277 n))) B).
Proof. exact erefl. Qed.

Lemma NADD_EQ_REFL f : nadd_eq f f.
Proof. unfold nadd_eq. exists 0. intro n. unfold dist; simpl. lia. Qed.

Lemma nadd_eq_sym f g : nadd_eq f g -> nadd_eq g f.
Proof. intros [b fg]. exists b. intro n. rewrite DIST_SYM. apply fg. Qed.

Lemma nadd_eq_trans f g h : nadd_eq f g -> nadd_eq g h -> nadd_eq f h.
Proof.
  intros [b fg] [c gh]. exists (b+c). intro n.
  apply (N.le_trans _ _ _ (@DIST_TRIANGLE _ (dest_nadd g n) _)).
  by apply N.add_le_mono.
Qed.

Add Relation _ nadd_eq
    reflexivity proved by NADD_EQ_REFL
    symmetry proved by nadd_eq_sym
    transitivity proved by nadd_eq_trans
as nadd_eq_rel.

Require Import Stdlib.Setoids.Setoid.

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

Lemma nadd_add_lcancel x y z : nadd_add x y = nadd_add x z -> y = z.
Proof.
  intro h. destruct x as [x hx]. destruct y as [y hy]. destruct z as [z hz].
  apply subset_eq_compat. unfold nadd_add in h. simpl in h. apply mk_inj in h.
  ext=>a. generalize (ext_fun h a); simpl; intro ha. lia.
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

Lemma nadd_inv_def : nadd_inv = (fun _23476 : nadd => @COND nadd (nadd_eq _23476 (nadd_of_num (NUMERAL N0))) (nadd_of_num (NUMERAL N0)) (mk_nadd (nadd_rinv _23476))).
Proof. exact erefl. Qed.

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

Lemma hreal_of_num_def : hreal_of_num = (fun m : N => mk_hreal (fun u : nadd => nadd_eq (nadd_of_num m) u)).
Proof. exact erefl. Qed.

Definition hreal_add : hreal -> hreal -> hreal := fun x : hreal => fun y : hreal => mk_hreal (fun u : nadd => exists x' : nadd, exists y' : nadd, (nadd_eq (nadd_add x' y') u) /\ ((dest_hreal x x') /\ (dest_hreal y y'))).

Lemma hreal_add_def : hreal_add = (fun x : hreal => fun y : hreal => mk_hreal (fun u : nadd => exists x' : nadd, exists y' : nadd, (nadd_eq (nadd_add x' y') u) /\ ((dest_hreal x x') /\ (dest_hreal y y')))).
Proof. exact erefl. Qed.

Lemma hreal_add_of_num p q :
  hreal_of_num (p + q) = hreal_add (hreal_of_num p) (hreal_of_num q).
Proof.
  unfold hreal_add, hreal_of_num. f_equal. ext=>x h.
  exists (nadd_of_num p). exists (nadd_of_num q). split.
  rewrite <- nadd_of_num_add. exact h. split.
  rewrite axiom_22_aux. 2: exists (nadd_of_num p); reflexivity. apply NADD_EQ_REFL.
  rewrite axiom_22_aux. 2: exists (nadd_of_num q); reflexivity. apply NADD_EQ_REFL.
  destruct h as [f [g [h1 [h2 h3]]]].
  rewrite axiom_22_aux in h2. 2: exists (nadd_of_num p); reflexivity.
  rewrite axiom_22_aux in h3. 2: exists (nadd_of_num q); reflexivity.
  rewrite nadd_of_num_add. rewrite h2 h3. exact h1.
Qed.

Lemma succ_eq_add_1 n : N.succ n = n + 1. Proof. lia. Qed.

Lemma hreal_of_num_S n : hreal_of_num (N.succ n) = hreal_add (hreal_of_num n) (hreal_of_num 1).
Proof. rewrite succ_eq_add_1 hreal_add_of_num. reflexivity. Qed.

Lemma hreal_add_sym p q : hreal_add p q = hreal_add q p.
Proof.
  unfold hreal_add. f_equal. ext=>x [y [z [h1 [h2 h3]]]].
  exists z. exists y. split. rewrite NADD_ADD_SYM. exact h1. auto.
  exists z. exists y. split. rewrite NADD_ADD_SYM. exact h1. auto.
Qed.

Lemma hreal_add_of_mk_hreal p q :
  hreal_add (mk_hreal (nadd_eq p)) (mk_hreal (nadd_eq q))
  = mk_hreal (nadd_eq (nadd_add p q)).
Proof.
  unfold hreal_add. apply f_equal. ext=> x h.

  unfold dest_hreal, mk_hreal in h. destruct h as [p' [q' [h1 [h2 h3]]]].
  rewrite dest_mk_aux_quotient in h2. 2: apply is_eq_class_of.
  rewrite dest_mk_aux_quotient in h3. 2: apply is_eq_class_of.
  rewrite h2 h3. exact h1.

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
  rewrite NADD_ADD_SYM (NADD_ADD_SYM (elt_of q)) in e.
  apply NADD_ADD_LCANCEL in e.
  f_equal. apply eq_class_intro. apply nadd_eq_sym. apply nadd_eq_trans.
  exact e.
Qed.

Definition hreal_mul : hreal -> hreal -> hreal := fun x : hreal => fun y : hreal => mk_hreal (fun u : nadd => exists x' : nadd, exists y' : nadd, (nadd_eq (nadd_mul x' y') u) /\ ((dest_hreal x x') /\ (dest_hreal y y'))).

Lemma hreal_mul_def : hreal_mul = (fun x : hreal => fun y : hreal => mk_hreal (fun u : nadd => exists x' : nadd, exists y' : nadd, (nadd_eq (nadd_mul x' y') u) /\ ((dest_hreal x x') /\ (dest_hreal y y')))).
Proof. exact erefl. Qed.

Definition hreal_le : hreal -> hreal -> Prop := fun x : hreal => fun y : hreal => @ε Prop (fun u : Prop => exists x' : nadd, exists y' : nadd, ((nadd_le x' y') = u) /\ ((dest_hreal x x') /\ (dest_hreal y y'))).

Lemma hreal_le_def : hreal_le = (fun x : hreal => fun y : hreal => @ε Prop (fun u : Prop => exists x' : nadd, exists y' : nadd, ((nadd_le x' y') = u) /\ ((dest_hreal x x') /\ (dest_hreal y y')))).
Proof. exact erefl. Qed.

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

Lemma hreal_inv_def : hreal_inv = (fun x : hreal => mk_hreal (fun u : nadd => exists x' : nadd, (nadd_eq (nadd_inv x') u) /\ (dest_hreal x x'))).
Proof. exact erefl. Qed.

(*****************************************************************************)
(* Operations on treal (pairs of hreal's). *)
(*****************************************************************************)

Definition treal_of_num : N -> prod hreal hreal := fun _23721 : N => @pair hreal hreal (hreal_of_num _23721) (hreal_of_num (NUMERAL N0)).

Lemma treal_of_num_def : treal_of_num = (fun _23721 : N => @pair hreal hreal (hreal_of_num _23721) (hreal_of_num (NUMERAL N0))).
Proof. exact erefl. Qed.

Definition treal_neg : (prod hreal hreal) -> prod hreal hreal := fun _23726 : prod hreal hreal => @pair hreal hreal (@snd hreal hreal _23726) (@fst hreal hreal _23726).

Lemma treal_neg_def : treal_neg = (fun _23726 : prod hreal hreal => @pair hreal hreal (@snd hreal hreal _23726) (@fst hreal hreal _23726)).
Proof. exact erefl. Qed.

Definition treal_add : (prod hreal hreal) -> (prod hreal hreal) -> prod hreal hreal := fun _23735 : prod hreal hreal => fun _23736 : prod hreal hreal => @pair hreal hreal (hreal_add (@fst hreal hreal _23735) (@fst hreal hreal _23736)) (hreal_add (@snd hreal hreal _23735) (@snd hreal hreal _23736)).

Lemma treal_add_def : treal_add = (fun _23735 : prod hreal hreal => fun _23736 : prod hreal hreal => @pair hreal hreal (hreal_add (@fst hreal hreal _23735) (@fst hreal hreal _23736)) (hreal_add (@snd hreal hreal _23735) (@snd hreal hreal _23736))).
Proof. exact erefl. Qed.

Lemma treal_add_of_num p q :
  treal_of_num (p + q) = treal_add (treal_of_num p) (treal_of_num q).
Proof.
  unfold treal_of_num, treal_add; simpl.
  f_equal; rewrite <- hreal_add_of_num; reflexivity.
Qed.

Lemma treal_add_sym  p q : treal_add p q = treal_add q p.
Proof. unfold treal_add. f_equal; apply hreal_add_sym. Qed.

Definition treal_mul : (prod hreal hreal) -> (prod hreal hreal) -> prod hreal hreal := fun _23757 : prod hreal hreal => fun _23758 : prod hreal hreal => @pair hreal hreal (hreal_add (hreal_mul (@fst hreal hreal _23757) (@fst hreal hreal _23758)) (hreal_mul (@snd hreal hreal _23757) (@snd hreal hreal _23758))) (hreal_add (hreal_mul (@fst hreal hreal _23757) (@snd hreal hreal _23758)) (hreal_mul (@snd hreal hreal _23757) (@fst hreal hreal _23758))).

Lemma treal_mul_def : treal_mul = (fun _23757 : prod hreal hreal => fun _23758 : prod hreal hreal => @pair hreal hreal (hreal_add (hreal_mul (@fst hreal hreal _23757) (@fst hreal hreal _23758)) (hreal_mul (@snd hreal hreal _23757) (@snd hreal hreal _23758))) (hreal_add (hreal_mul (@fst hreal hreal _23757) (@snd hreal hreal _23758)) (hreal_mul (@snd hreal hreal _23757) (@fst hreal hreal _23758)))).
Proof. exact erefl. Qed.

Definition treal_le : (prod hreal hreal) -> (prod hreal hreal) -> Prop := fun _23779 : prod hreal hreal => fun _23780 : prod hreal hreal => hreal_le (hreal_add (@fst hreal hreal _23779) (@snd hreal hreal _23780)) (hreal_add (@fst hreal hreal _23780) (@snd hreal hreal _23779)).

Lemma treal_le_def : treal_le = (fun _23779 : prod hreal hreal => fun _23780 : prod hreal hreal => hreal_le (hreal_add (@fst hreal hreal _23779) (@snd hreal hreal _23780)) (hreal_add (@fst hreal hreal _23780) (@snd hreal hreal _23779))).
Proof. exact erefl. Qed.

(*Lemma treal_le_refl x : treal_le x x.
Proof.
  unfold treal_le. destruct x as [x1 x2]. simpl. apply hreal_le_refl.
Qed.

Add Relation _ treal_le
    reflexivity proved by treal_le_refl
    (*transitivity proved by treal_le_trans*)
as treal_le_rel.*)

Definition treal_inv : (prod hreal hreal) -> prod hreal hreal := fun _23801 : prod hreal hreal => @COND (prod hreal hreal) ((@fst hreal hreal _23801) = (@snd hreal hreal _23801)) (@pair hreal hreal (hreal_of_num (NUMERAL N0)) (hreal_of_num (NUMERAL N0))) (@COND (prod hreal hreal) (hreal_le (@snd hreal hreal _23801) (@fst hreal hreal _23801)) (@pair hreal hreal (hreal_inv (@ε hreal (fun d : hreal => (@fst hreal hreal _23801) = (hreal_add (@snd hreal hreal _23801) d)))) (hreal_of_num (NUMERAL N0))) (@pair hreal hreal (hreal_of_num (NUMERAL N0)) (hreal_inv (@ε hreal (fun d : hreal => (@snd hreal hreal _23801) = (hreal_add (@fst hreal hreal _23801) d)))))).

Lemma treal_inv_def : treal_inv = (fun _23801 : prod hreal hreal => @COND (prod hreal hreal) ((@fst hreal hreal _23801) = (@snd hreal hreal _23801)) (@pair hreal hreal (hreal_of_num (NUMERAL N0)) (hreal_of_num (NUMERAL N0))) (@COND (prod hreal hreal) (hreal_le (@snd hreal hreal _23801) (@fst hreal hreal _23801)) (@pair hreal hreal (hreal_inv (@ε hreal (fun d : hreal => (@fst hreal hreal _23801) = (hreal_add (@snd hreal hreal _23801) d)))) (hreal_of_num (NUMERAL N0))) (@pair hreal hreal (hreal_of_num (NUMERAL N0)) (hreal_inv (@ε hreal (fun d : hreal => (@snd hreal hreal _23801) = (hreal_add (@fst hreal hreal _23801) d))))))).
Proof. exact erefl. Qed.

Definition treal_eq : (prod hreal hreal) -> (prod hreal hreal) -> Prop := fun _23810 : prod hreal hreal => fun _23811 : prod hreal hreal => (hreal_add (@fst hreal hreal _23810) (@snd hreal hreal _23811)) = (hreal_add (@fst hreal hreal _23811) (@snd hreal hreal _23810)).

Lemma treal_eq_def : treal_eq = (fun _23810 : prod hreal hreal => fun _23811 : prod hreal hreal => (hreal_add (@fst hreal hreal _23810) (@snd hreal hreal _23811)) = (hreal_add (@fst hreal hreal _23811) (@snd hreal hreal _23810))).
Proof. exact erefl. Qed.

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
  rewrite (hreal_add_sym z2). rewrite xy yz.

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

Module Export HL.

Definition mk_real := mk_quotient treal_eq.
Definition dest_real := dest_quotient treal_eq.

Lemma axiom_23 : forall (a : real), (mk_real (dest_real a)) = a.
Proof. exact (mk_dest_quotient treal_eq). Qed.

Lemma axiom_24_aux : forall r, (exists x, r = treal_eq x) -> dest_real (mk_real r) = r.
Proof. exact (dest_mk_aux_quotient treal_eq). Qed.

Lemma axiom_24 : forall (r : (prod hreal hreal) -> Prop), ((fun s : (prod hreal hreal) -> Prop => exists x : prod hreal hreal, s = (treal_eq x)) r) = ((dest_real (mk_real r)) = r).
Proof. exact (dest_mk_quotient treal_eq). Qed.

End HL.
