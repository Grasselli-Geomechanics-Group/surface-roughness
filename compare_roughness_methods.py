import matplotlib.pyplot as plt

from surface_roughness import Surface

s = Surface("example_surface.stl")

s.evaluate_delta_t()
delta_t_az = s.delta_t("az")
delta_t = s.delta_t("delta_t")
deltastar_t = s.delta_t("delta*_t")

delta_a_az = s.delta_a("az")
delta_a = s.delta_a("delta_a")
deltastar_a = s.delta_a("delta*_a")

# delta_n_az = s.delta_n("az")
# delta_n = s.delta_n("delta_n")
# deltastar_n = s.delta_n("delta*_n")

thetacp1_az = s.thetamax_cp1("az")
thetacp1 = s.thetamax_cp1("thetamax_cp1")

meandip_az = s.meandip("az")
meandip = s.meandip("mean_dip")
stddip = s.meandip("std_dip")

plt.polar(delta_t_az,delta_t,'b')
plt.polar(delta_t_az,deltastar_t,'b--')
plt.polar(delta_a_az,delta_a,'r')
plt.polar(delta_a_az,deltastar_a,'r--')
# plt.polar(delta_n_az,delta_n)
# plt.polar(delta_n_az,deltastar_n)
plt.polar(thetacp1_az,thetacp1,'k')
plt.polar(meandip_az,meandip,'g')
plt.polar(meandip_az,stddip,'g--')
plt.legend([
    r'$\Delta_T$',r'$\Delta^*_T$',
    r'$\Delta_A$',r'$\Delta^*_A$',
    r'$\theta^*_{max}/(C+1)$',
    'Mean apparent dip','Std. dev. apparent dip'],
    loc="upper right",bbox_to_anchor=(1.65,0.75))
plt.tight_layout()
plt.show()